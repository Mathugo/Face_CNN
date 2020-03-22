from get_faces import *
from process_to_h5 import *

def main():
    face = GetFaces()
    face.run_video()
    face.run_img()

main()