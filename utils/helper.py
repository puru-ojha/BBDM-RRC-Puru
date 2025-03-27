import matplotlib.pyplot as plt
import numpy as np
import cv2

input_path = "../sample_to_eval/condition/image_827.png"
output_path = "../sample_to_eval/200/image_827/output_0.png"
gt_path = "../sample_to_eval/ground_truth/image_827.png"

input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
output_img = cv2.cvtColor(cv2.imread(output_path), cv2.COLOR_BGR2RGB)
gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

titles = ["Input", "Output", "GT"]
images = [input_img, output_img, gt_img]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()

plt.savefig("../slack-results/figure_3.png", dpi=300, bbox_inches='tight')

# plt.show()
