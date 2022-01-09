#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json
import boto3
import time
import os
from deepspeech import Model#, printVersions
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)

def words_from_metadata(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i in range(0, metadata.num_items):
        item = metadata.items[i]
        # Append character to word if it's not a space
        if item.character != " ":
            word = word + item.character
        # Word boundary is either a space or the last character in the array
        if item.character == " " or i == metadata.num_items - 1:
            word_duration = item.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time "] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0
        else:
            if len(word) == 1:
                # Log the start time of the new word
                word_start_time = item.start_time

    return word_list


def metadata_json_output(metadata):
    json_result = dict()
    json_result["words"] = words_from_metadata(metadata)
    json_result["confidence"] = metadata.confidence
    return json.dumps(json_result)
	


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        #printVersions()
        exit(0)
my_dir_set = set(os.listdir('/tmp/'))
s3_client = boto3.resource('s3')
bucket_name = "deepspeech-package-new"
#file_names=["240K.wav","480K.wav","1020K.wav","output_graph.pbmm"]
file_names=["output_graph.pbmm"]
file_size = [12, 24, 28, 32, 36, 42, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 137, 140, 144]

for val in file_size: #(12,120,4):
    file_names.append("{}K.wav".format(val))

for file_name in file_names:
    if file_name in my_dir_set:
        print("file exists")
        continue
    print(file_name)
    s3_client.Bucket(bucket_name).download_file(file_name,'/tmp/{}'.format(file_name))
    print("downloaded file {}".format(file_name))

model_path = '/tmp/output_graph.pbmm'
print('Loading model from file {}'.format(model_path))
model_load_start = timer()
ds = Model('/tmp/output_graph.pbmm')#, args.beam_width)
model_load_end = timer() - model_load_start
print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

def lambda_handler(event, context):
    batch_size = event["batch_size"]
    file_name = "{}K.wav".format(event["file_name"])
    infer_time = main(batch_size, file_name)
    return infer_time

def main(batch_size, file_name):

    desired_sample_rate = ds.sampleRate()

    fin = wave.open('/tmp/{}'.format(file_name), 'rb')
    fs = fin.getframerate()
    if fs != desired_sample_rate:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs, desired_sample_rate), file=sys.stderr)
        fs, audio = convert_samplerate(args.audio, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs)
    fin.close()

    print('Running inference.', file=sys.stderr)
    inference_start = int(time.time()*1000)#timer()
    for i in range(batch_size):
        ds.stt(audio)
    print(ds.stt(audio))
    inference_end = int(time.time()*1000) - inference_start#timer() - inference_start
    print('Inference took %0.3fms for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
    return inference_end

if __name__ == '__main__':
   print("This is main")
   #main()
