import os
import tensorflow as tf
from tensorflow import keras
from net_v6 import BidirectionalRestorer_V6
from meg_2_tf import load_from_meg
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def parse_args():
    parser = argparse.ArgumentParser(description='to get tflite')
    parser.add_argument("-n", default=False, action='store_true', help="enable None input shape")
    parser.add_argument('--mgepath', default = None, type=str, help='the path to xxx.mge pretrained model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.n:
        input_shape = (1, None, None, 30)
        save_path = "model_none.tflite"
    else:
        input_shape = (1, 180, 320, 30)
        save_path = "model.tflite"

    netG = BidirectionalRestorer_V6()

    print("building ....")
    netG.build(input_shape)
    print("build ok!!")
    netG.summary()

    netG.compute_output_shape(input_shape=input_shape)

    netG = load_from_meg(netG, args.mgepath)
    
    # inputs1 = np.zeros((10, 18, 32, 30))
    # netG.predict(inputs1, batch_size = 1, verbose= 1)
    print("testing...")
    inputs2 = np.zeros((5, 180, 320, 30))
    netG.predict(inputs2, batch_size = 1, verbose= 1)


    print("converting to tflite....")
    converter = tf.lite.TFLiteConverter.from_keras_model(netG)
    converter.experimental_new_converter = True
    tflite_model = converter.convert()
    
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    print("done!")
