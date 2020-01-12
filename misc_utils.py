import random
import io
import pprint

import numpy as np
import absl.flags
from absl import logging
import tensorflow as tf
from PIL import Image
from gym import wrappers


class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self


def define_flags_with_default(**kwargs):
	for key, val in kwargs.items():
		if isinstance(val, bool):
			# Note that True and False are instances of int.
			absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
		elif isinstance(val, int):
			absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
		elif isinstance(val, float):
			absl.flags.DEFINE_float(key, val, 'automatically defined flag')
		elif isinstance(val, str):
			absl.flags.DEFINE_string(key, val, 'automatically defined flag')
		else:
			raise ValueError('Incorrect value type')
	return kwargs


def set_random_seed(seed):
	np.random.seed(seed)
	tf.random.set_random_seed(seed)
	random.seed(seed)


import os
import csv


def direct_logging(data, output_dir):
	for metric in data:
		metric_dir = output_dir + metric
		if os.path.isdir(metric_dir) != True:
			os.makedirs(metric_dir, exist_ok=True)
		
		with open(metric_dir + '/progress.csv', 'a') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerow([data[metric]])
		csvFile.close()


class TensorBoardLogger(object):
	"""Logging to TensorBoard outside of TensorFlow ops."""
	
	def __init__(self, output_dir):
		if not tf.gfile.Exists(output_dir):
			tf.gfile.MakeDirs(output_dir)
		self.output_dir = output_dir
		self.file_writer = tf.summary.FileWriter(output_dir)
	
	def log_scaler(self, step, name, value):
		summary = tf.Summary(
			value=[tf.Summary.Value(tag=name, simple_value=value)]
		)
		self.file_writer.add_summary(summary, step)
	
	def log_image(self, step, name, image):
		summary = tf.Summary(
			value=[tf.Summary.Value(
				tag=name,
				image=self._make_image(image)
			)]
		)
		self.file_writer.add_summary(summary, step)
	
	def log_images(self, step, data):
		if len(data) == 0:
			return
		summary = tf.Summary(
			value=[
				tf.Summary.Value(tag=name, image=self._make_image(image))
				for name, image in data.items() if image is not None
			]
		)
		self.file_writer.add_summary(summary, step)
	
	def _make_image(self, tensor):
		"""Convert an numpy representation image to Image protobuf"""
		height, width, channel = tensor.shape
		image = Image.fromarray(tensor)
		output = io.BytesIO()
		image.save(output, format='PNG')
		image_string = output.getvalue()
		output.close()
		return tf.Summary.Image(
			height=height,
			width=width,
			colorspace=channel,
			encoded_image_string=image_string
		)
	
	def log_dict(self, step, data):
		
		summary = tf.Summary(
			value=[
				tf.Summary.Value(tag=name, simple_value=value)
				for name, value in data.items() if value is not None
			]
		)
		direct_logging(data, self.output_dir + 'logs/')
		self.file_writer.add_summary(summary, step)
	
	def flush(self):
		self.file_writer.flush()


def unwrapped_env(env):
	if isinstance(env, wrappers.TimeLimit) \
		or isinstance(env, wrappers.Monitor) \
		or isinstance(env, wrappers.FlattenDictWrapper):
		return env.unwrapped
	return env


def average_metrics(metrics):
	if len(metrics) == 0:
		return {}
	new_metrics = {}
	for key in metrics[0].keys():
		new_metrics[key] = np.mean([m[key] for m in metrics])
	
	return new_metrics


def print_flags(flags, flags_def):
	logging.info(
		'Running training with hyperparameters: \n{}'.format(
			pprint.pformat(
				['{}: {}'.format(key, getattr(flags, key)) for key in flags_def]
			)
		)
	)


def parse_network_arch(arch):
	if len(arch) == 0:
		return []
	return [int(x) for x in arch.split('-')]
