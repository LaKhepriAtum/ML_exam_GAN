import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
from IPython.core.pylabtools import figsize

tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector() # 얼굴을 찾아주는 모델
shape = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat') # tensorflow의 모델을 keras와 달리 여러개
# 얼굴의 landmark를 찾아서 list로
# img = dlib.load_rgb_image('./imgs/09.jpg')
#
# plt.figure(figsize = (16,10))
# plt.imshow(img)
# plt.show()
#
# img_result = img.copy() # 기존의 img를 가져와 img_result에 복사
# dets = detector(img, 1) # upsamle 계수, -> data 가 적을 때 data의 수를 증가시키는 것
#
# if len(dets) == 0: # 얼굴을 못 찾았으면
#     print('Not find faces')
# else:
#     fig, ax = plt.subplots(1, figsize = (10, 16))
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height() #얼굴의 x좌표, 넓이 높이 를 찾는다.
#         rect = patches.Rectangle((x, y), w, h, # 얼굴의 x, y 좌표를 찍고 사각형을 그린다
#                                  linewidth=2, edgecolor='b', facecolor='None') #facecolor='None'-> 얼굴을 덮지 않는다
#         ax.add_patch(rect)
# ax.imshow(img_result)
# plt.show()
#
# fig , ax = plt.subplots(1, figsize=(16,10))
# obj = dlib.full_object_detections() #
#
# for detection in dets:
#     s = shape(img, detection)
#     obj.append(s)
#
#     for point in s.parts(): # 68개의 점을 좌표로
#         circle = patches.Circle((point.x, point.y), #patches-> 도형을 그려주는 함수, point.x, point.y 좌표
#                                 radius=3, edgecolor='b', facecolor='b')
#         ax.add_patch(circle)
#         ax.imshow(img_result)
# plt.show()

def align_faces(img):
    dets = detector(img, 1) #이미지에서 얼굴 찾기
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape(img, detection) # landmark 찾기
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size= 256#get_face_chips로 얼굴들 짤라서 다 찾아서 return, 256-> 한 사진당 크기
                                , padding = 0.35)#padding-> 얼굴을 조금 넉넉하게 짜른다
    return faces
# test_img = dlib.load_rgb_image('./imgs/02.jpg')
# test_faces = align_faces(test_img)
# fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(10,8)) # 1행의 test_faces+1개의 열을 만든다,+1-> 원본
# axes[0].imshow(test_img) # 원본
# for i, face in enumerate(test_faces):
#     axes[i +1].imshow(face)
# plt.show()

sess = tf.Session()

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)

saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train. latest_checkpoint('./models')) # 모델 불러오기
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0') #
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0') # generator 불러오기

def preprocess(img):
    return img / 127.5 -1 # 스케일링
def deprocess(img):
    return (img +1) / 2 # 디스케일링

img1 = dlib.load_rgb_image('./imgs/no_makeup/xfsy_0226.png')
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('./imgs/makeup/XMY-136.png')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()

src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0) # reshape를 해준다

Y_img = preprocess(ref_img) # 스케일링 해서 넣고
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict = {X:X_img, Y:Y_img}) #Xs 모델(generator)을 run 한다.
output_img = deprocess(output[0]) # 디스케일링 한다

fig, axes = plt.subplots(1, 3, figsize=(8, 5))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
axes[2].imshow(output_img)
plt.show()
