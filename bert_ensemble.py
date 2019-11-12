#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization


# In[3]:


def pretty_print(result):
    df = pd.DataFrame([result]).T
    df.columns = ["values"]
    return df


# In[4]:


def create_tokenizer_from_hub_module(bert_model_hub):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info",
                                        as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"]
            ])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)


def make_features(dataset, label_list, MAX_SEQ_LENGTH, tokenizer, DATA_COLUMN,
                  LABEL_COLUMN):
    input_example = dataset.apply(lambda x: bert.run_classifier.InputExample(
        guid=None, text_a=x[DATA_COLUMN], text_b=None, label=x[LABEL_COLUMN]),
                                  axis=1)
    features = bert.run_classifier.convert_examples_to_features(
        input_example, label_list, MAX_SEQ_LENGTH, tokenizer)
    return features


# creating model, replace it by custom model
def create_model(bert_model_hub, is_predicting, input_ids, input_mask,
                 segment_ids, labels, num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(bert_model_hub, trainable=True)
    bert_inputs = dict(input_ids=input_ids,
                       input_mask=input_mask,
                       segment_ids=segment_ids)
    bert_outputs = bert_module(inputs=bert_inputs,
                               signature="tokens",
                               as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels],
                                  initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(
            tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(bert_model_hub, num_labels, learning_rate,
                     num_train_steps, num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            (loss, predicted_labels,
             log_probs) = create_model(bert_model_hub, is_predicting,
                                       input_ids, input_mask, segment_ids,
                                       label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(loss,
                                                          learning_rate,
                                                          num_train_steps,
                                                          num_warmup_steps,
                                                          use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(label_ids,
                                                       predicted_labels)
                auc = tf.metrics.auc(label_ids, predicted_labels)
                recall = tf.metrics.recall(label_ids, predicted_labels)
                precision = tf.metrics.precision(label_ids, predicted_labels)
                true_pos = tf.metrics.true_positives(label_ids,
                                                     predicted_labels)
                true_neg = tf.metrics.true_negatives(label_ids,
                                                     predicted_labels)
                false_pos = tf.metrics.false_positives(label_ids,
                                                       predicted_labels)
                false_neg = tf.metrics.false_negatives(label_ids,
                                                       predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels,
             log_probs) = create_model(bert_model_hub, is_predicting,
                                       input_ids, input_mask, segment_ids,
                                       label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


def estimator_builder(bert_model_hub, OUTPUT_DIR, SAVE_SUMMARY_STEPS,
                      SAVE_CHECKPOINTS_STEPS, label_list, LEARNING_RATE,
                      num_train_steps, num_warmup_steps, BATCH_SIZE):

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(bert_model_hub=bert_model_hub,
                                num_labels=len(label_list),
                                learning_rate=LEARNING_RATE,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params={"batch_size": BATCH_SIZE})
    return estimator, model_fn, run_config


# In[5]:


def run_on_dfs(
        train,
        test,
        DATA_COLUMN,
        LABEL_COLUMN,
        MAX_SEQ_LENGTH=128,
        BATCH_SIZE=32,
        LEARNING_RATE=2e-5,
        NUM_TRAIN_EPOCHS=3.0,
        WARMUP_PROPORTION=0.1,
        SAVE_SUMMARY_STEPS=100,
        SAVE_CHECKPOINTS_STEPS=10000,
        bert_model_hub="https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
):

    label_list = train[LABEL_COLUMN].unique().tolist()

    tokenizer = create_tokenizer_from_hub_module(bert_model_hub)

    train_features = make_features(train, label_list, MAX_SEQ_LENGTH,
                                   tokenizer, DATA_COLUMN, LABEL_COLUMN)
    test_features = make_features(test, label_list, MAX_SEQ_LENGTH, tokenizer,
                                  DATA_COLUMN, LABEL_COLUMN)

    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    estimator, model_fn, run_config = estimator_builder(
        bert_model_hub, OUTPUT_DIR, SAVE_SUMMARY_STEPS, SAVE_CHECKPOINTS_STEPS,
        label_list, LEARNING_RATE, num_train_steps, num_warmup_steps,
        BATCH_SIZE)

    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    test_input_fn = run_classifier.input_fn_builder(features=test_features,
                                                    seq_length=MAX_SEQ_LENGTH,
                                                    is_training=False,
                                                    drop_remainder=False)

    result_dict = estimator.evaluate(input_fn=test_input_fn, steps=None)
    return result_dict, estimator


# In[6]:


import random
random.seed(10)


# In[7]:


OUTPUT_DIR = 'output'


# In[8]:


dataset = pd.read_csv('./data/train.csv')
train_data, test = train_test_split(dataset, test_size=0.1, random_state=0)

train_data_pos = train_data.loc[train_data['label'] == 1]
train_data_neg = train_data.loc[train_data['label'] == 0]

train_set = np.array_split(train_data_neg, 6)
for item in train_set:
    item = item.append(train_data_pos)



# In[9]:


myparam = {
    "DATA_COLUMN": "comment",
    "LABEL_COLUMN": "label",
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 2e-5,
    "NUM_TRAIN_EPOCHS": 3,
    "bert_model_hub": "/Users/kevin/Workspace/food_safety/model/bert_base"
}


# In[10]:

result_set = []
estimator_set = []
for item in train_set:
    result, estimator = run_on_dfs(item, test, **myparam)
    result_set.append(result)
    estimator_set.append(estimator)


# In[ ]:


pretty_print(result_set)


# In[ ]:


def getPrediction(in_sentences, estimator):
    labels = ["Negaitive", "Positive"]
    label_list = train['label'].unique().tolist()
    MAX_SEQ_LENGTH = 128
    bert_model_hub = "/Users/kevin/Workspace/food_safety/model/bert_base"

    tokenizer = create_tokenizer_from_hub_module(bert_model_hub)
    input_examples = [
        bert.run_classifier.InputExample(guid=None,
                                         text_a=x,
                                         text_b=None,
                                         label=0) for x in in_sentences
    ]
    input_features = bert.run_classifier.convert_examples_to_features(
        input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = bert.run_classifier.input_fn_builder(
        features=input_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    predictions = estimator.predict(predict_input_fn)

    return [(sentence, prediction['probabilities'],
             labels[prediction['labels']])
            for sentence, prediction in zip(in_sentences, predictions)]


# In[ ]:


target = pd.read_csv('test_new.csv')
pred_sentences = target['comment'].values.tolist()
predictions_set = []
for estimator in estimator_set:
    predictions = getPrediction(pred_sentences, estimator)
    predictions_set.append(predictions)


# In[ ]:


predictions_set


# In[ ]:


target['count'] = 0
for prediction in predictions_set:
    for row in range(len(predictions[0])):
        if predictions[row][2] == 'Positive':
            target.loc[row, 'count'] += 1

target['label'] = None
for index, row in target.iterrows():
    if row['count'] > 2:
        row['label'] = 1
    else:
        row['label'] = 0


# In[ ]:


target.head()


# In[ ]:


target_o = target[['id', 'label']]
target_o.to_csv('./data/Predicts/output_BERT_ensemble.csv', index=False)
