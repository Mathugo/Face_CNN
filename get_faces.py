# -*- coding: utf-8 -*-
from imutils import paths
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import argparse
import os
from process_to_h5 import *
import tempfile

class GetFaces:

    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-d", "--dataset", type=str, required=True,
        help="add dataset path with images inside or video")
        ap.add_argument("-o", "--output", type=str, required=True,
        help="output folder")
        self.args = vars(ap.parse_args())
        self.output = self.args["output"]
        self.dataset = self.args["dataset"]
        self.nbVideo=2
        print("[*] Configuring MTCNN ..")
        self.mtcnn = MTCNN(margin=30, select_largest=False, post_process=False, device='cuda:0')
        print("[*] Done")
        self.imagePaths = list(paths.list_images(self.dataset))
        self.filespaths = list(paths.list_files(self.dataset))

    def run_video(self):
        for name in self.filespaths:
            if ".mp4" in name:
                self.video(name)
                self.nbVideo+=1

    def run_img(self):
        i = 1
        for name in self.imagePaths:
            print("[*] Processing image {}/{} -> {}".format(i, len(self.imagePaths), name))
            output_name = self.output+"\\{}.jpg".format(i)
            i+=1
            self.getFace(output_name, input_name=name)

    def proc_to_h5(self, filename):
        proc1 = proc_h5()
        output = tempfile.NamedTemporaryFile()
        output = output.name.split("\\")
        output = output[len(output)-1]+".hdf5" 
        output = "output\\hdf5\\"+output
        proc1.jpg_to_h5(filename, output)

    def getFace(self, output_name, input_name=False, frame=False):
        if frame is False and input_name is not False:
            frame = cv2.imread(input_name)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face = self.mtcnn(frame)
        plt.figure(figsize=(12, 8))
        
        try:
            plt.imshow(face.permute(1, 2, 0).int().numpy())
            plt.axis('off')
            plt.savefig(output_name)
            #print("[*] Converting .jpg to h5py file")
            #self.proc_to_h5(output_name)
            #print("[*] Done")
        except:
            print("[!] Cannot detect face on {}".format(output_name))
        plt.close('all')

    def get_count_frame(self, path):
        video = cv2.VideoCapture(path)
        total=0
        success, frame = video.read()
        while success:
            total+=1
            success, frame = video.read()
        return total

    def video(self, name):
        print("[*] Getting number of frames of {} ...".format(name))
        total_frame = self.get_count_frame(name)
        print("[!] {} frames to process".format(total_frame))

        v_cap = cv2.VideoCapture(name)
        success, frame_ = v_cap.read()
        i=1

        while success:
            print("Processing video {}/{}".format(i, total_frame))
            output_name = self.output+"\\Video{}_{}.jpg".format(self.nbVideo, i)
            i+=1
            self.getFace(output_name, frame=frame_)
            success, frame_ = v_cap.read()

