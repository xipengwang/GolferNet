import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(sys.argv[1])
num_of_frames = 3
i = 0
while(cap.isOpened()):
  ret, frame = cap.read()
  crop_img = frame[:, 210:-210] # 540x540
  print(crop_img.shape)
  crop_img = crop_img[14:-14, 14:-14] # 512x512
  print(crop_img.shape)
  plt.imshow(crop_img)
  plt.show()
  cv2.imwrite(f'./data/POSE/{i}.png', crop_img)
  if i == num_of_frames:
      break
  i += 1
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

# (110, 317)
# (105, 306)
# (102, 296)
# (100, 286)
