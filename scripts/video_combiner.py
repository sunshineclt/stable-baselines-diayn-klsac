import cv2
import numpy as np
from tqdm import tqdm


def convert(skill):
    def add_text(frame, text):
        cv2.putText(frame, text, (180, 500), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255))

    def add_text_all():
        add_text(frameLeftUp, "1M")
        add_text(frameRightUp, "1.2M")
        add_text(frameLeftDown, "1.4M")
        add_text(frameRightDown, "1.6M")

    videoLeftUp = cv2.VideoCapture('./log/diayn/HalfCheetah-v2_5/videos_1000000/skill_%d.avi' % skill)
    videoRightUp = cv2.VideoCapture('./log/diayn/HalfCheetah-v2_5/videos_1200000/skill_%d.avi' % skill)
    videoLeftDown = cv2.VideoCapture('./log/diayn/HalfCheetah-v2_5/videos_1400000/skill_%d.avi' % skill)
    videoRightDown = cv2.VideoCapture('./log/diayn/HalfCheetah-v2_5/videos_1600000/skill_%d.avi' % skill)

    fps = videoLeftUp.get(cv2.CAP_PROP_FPS)

    width = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(videoLeftUp.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter('./log/diayn/HalfCheetah-v2_5/videos_combined/skill_%d.mp4' % skill,
                                  cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

    successLeftUp, frameLeftUp = videoLeftUp.read()
    successLeftDown, frameLeftDown = videoLeftDown.read()
    successRightUp, frameRightUp = videoRightUp.read()
    successRightDown, frameRightDown = videoRightDown.read()
    add_text_all()


    while successLeftUp and successLeftDown and successRightUp and successRightDown:
        frameLeftUp = cv2.resize(frameLeftUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameLeftDown = cv2.resize(frameLeftDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameRightUp = cv2.resize(frameRightUp, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
        frameRightDown = cv2.resize(frameRightDown, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)

        frameUp = np.hstack((frameLeftUp, frameRightUp))
        frameDown = np.hstack((frameLeftDown, frameRightDown))
        frame = np.vstack((frameUp, frameDown))

        videoWriter.write(frame)
        successLeftUp, frameLeftUp = videoLeftUp.read()
        successLeftDown, frameLeftDown = videoLeftDown.read()
        successRightUp, frameRightUp = videoRightUp.read()
        successRightDown, frameRightDown = videoRightDown.read()
        add_text_all()

    videoWriter.release()
    videoLeftUp.release()
    videoLeftDown.release()
    videoRightUp.release()
    videoRightDown.release()


for skill in tqdm(range(20)):
    convert(skill)
