import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pylab import *
from torchvision import transforms


def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

'''
def showTestResult(img, groundtruth, predict):
    # red is predict, green is groundtruth
    a = plt.imshow(tensor_to_PIL(img))
    groundtruth1 = groundtruth.cpu().detach.numpy()
    predict1 = predict.cpu().detach.numpy()
    plot(groundtruth1, 'g*')
    plot(predict1, 'r*')
    a.show()
'''

def t_est1(args, model, device, test_loader ):
    print("FINISH")
    model.eval()
    batch_idx = 0
    test_loss = 0.0
    #sum = 0.0
    d = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data, target = batch['image'].to(device).float(), batch['landmarks'].to(device).float()
            output = model(data)
            test_loss += F.l1_loss(output, target).item() # sum up batch loss

            if batch_idx % 1 == 0:
                #showTestResult(batch['image'][0], batch['landmarks'], output)
                i = output.cpu().detach().numpy()
                j = batch['landmarks'].cpu().detach().numpy()
                plt.imshow(tensor_to_PIL(batch['image'][0]))
                plt.xlim(0, 128)
                plt.ylim(64, 0)
                plt.plot([i[0][0], i[0][2]], [i[0][1], i[0][3]], color='r')
                plt.plot([i[0][2], i[0][4]], [i[0][3], i[0][5]], color='r')
                plt.plot([i[0][4], i[0][6]], [i[0][5], i[0][7]], color='r')
                plt.plot([i[0][6], i[0][0]], [i[0][7], i[0][1]], color='r')
                plt.scatter(j[0][0], j[0][1], color = 'pink')
                plt.scatter(j[0][2], j[0][3], color = 'pink')
                plt.scatter(j[0][4], j[0][5], color = 'pink')
                plt.scatter(j[0][6], j[0][7], color = 'pink')
                d1 = sqrt(square(i[0][0] - j[0][0]) + square(i[0][1] - j[0][1]))
                d2 = sqrt(square(i[0][2] - j[0][2]) + square(i[0][3] - j[0][3]))
                d3 = sqrt(square(i[0][4] - j[0][4]) + square(i[0][5] - j[0][5]))
                d4 = sqrt(square(i[0][6] - j[0][6]) + square(i[0][7] - j[0][7]))
                d.append(float((d1 + d2 + d3 + d4)/4))
                plt.show()
                # print('evaluate value:', d[batch_idx] / 128)
                # sum = sum()/128
                # print('Average distance loss:',d)
    #        tim.sleep(3)
    #             plt.show()
    print('Average evaluate:',sum(d) / (128*batch_idx))
    # print('average evaluate value:', sum / batch_idx)
    print('test loss:', test_loss/batch_idx)
