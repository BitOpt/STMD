import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def ndarray_show(img_name: str, input_ndarray):
    max_value = np.max(input_ndarray)
    min_value = np.min(input_ndarray)

    plot_nor = (input_ndarray - min_value) / (max_value - min_value) * 255
    cv2.imshow(img_name, cv2.applyColorMap(plot_nor.astype(np.uint8), cv2.COLORMAP_JET))
    cv2.waitKey(1)


def tensor_show(img_name: str, input_tensor):
    plot = input_tensor.clone().squeeze().cpu().numpy()
    plot_downsample = cv2.resize(plot, (1280, 720), interpolation=cv2.INTER_LINEAR)
    min = plot_downsample.min()
    max = plot_downsample.max()
    plot_nor = (plot_downsample - min) / (max - min) * 255
    cv2.imshow(img_name, cv2.applyColorMap(plot_nor.astype(np.uint8), cv2.COLORMAP_JET))
    cv2.waitKey(1)


def ndarray_matplotlibshow(img_name: str, input_array):
    fig, ax = plt.subplots()
    cax = ax.imshow(input_array, cmap='viridis', interpolation='nearest')
    fig.colorbar(cax)
    ax.set_title(img_name)
    plt.show()
    plt.close(fig)


def tensor_matplotlibshow(img_name: str, input_tensor):
    plot = input_tensor.clone().squeeze().cpu().numpy()
    fig, ax = plt.subplots()
    cax = ax.imshow(plot, cmap='viridis', interpolation='nearest')
    fig.colorbar(cax)
    ax.set_title(img_name)
    plt.show()
    plt.close(fig)


def ndarray_multi_matplotlibshow(img_name_list: list, input_array_list: list):
    fig_list = [plt.subplots() * len(img_name_list)]
    for i, element in enumerate(img_name_list):
        fig, ax = fig_list[i]
        cax = ax.imshow(input_array_list[i], cmap='viridis', interpolation='nearest')
        fig.colorbar(cax)
        ax.set_title(img_name_list[i])
    plt.show()
    plt.close('all')


def tensor_save(img_name: str, input_tensor):
    plot = input_tensor.clone().squeeze().cpu().numpy()
    min = plot.min()
    max = plot.max()
    plot_nor = (plot - min) / (max - min) * 255
    cv2.imwrite(img_name, cv2.applyColorMap(plot_nor[1032:1132, 592:1092].astype(np.uint8), cv2.COLORMAP_JET))


def ndarray_save(img_name: str, input_ndarray):
    cv2.imwrite(img_name, input_ndarray)


def tensor_save_txt(txt_name: str, input_tensor):
    if isinstance(input_tensor, np.ndarray) is False:
        input_tensor = input_tensor.clone().squeeze().cpu().numpy()
    rows, cols = input_tensor.shape
    with open(txt_name, 'w') as f:
        for row in range(rows):
            f.write(str(int(input_tensor[row][0])) + ' ' + str(int(input_tensor[row][1])) + ' ' + str(
                input_tensor[row][2]) + '\n')


def tensor_save_txt_row(txt_name: str, input_tensor):
    if isinstance(input_tensor, np.ndarray) is False:
        input_tensor = input_tensor.clone().squeeze().cpu().numpy()
    rows, cols = input_tensor.shape
    # avg_value = np.mean(input_tensor[1080:1090], axis=0)

    with open(txt_name, 'w') as f:
        for col in np.arange(592, 1092, 1):
            # if col == 500:
            #     break
            f.write(str(1082) + ' ' + str(int(col)) + ' ' + str(
                input_tensor[1082][col]) + '\n')


def tensor_save_txt_xy(txt_name: str, input_tensor):
    if isinstance(input_tensor, np.ndarray) is False:
        input_tensor = input_tensor.clone().squeeze().cpu().numpy()
    np.savetxt(txt_name, input_tensor[1060:2001, 0:501], delimiter=' ', fmt='%.5f')


def lptc_save(img_name: str, lptc_coordinates, lptc_values):
    points = lptc_coordinates[:, 2:4]
    values = np.array(lptc_values)

    values_nor = (values - np.min(values)) / (np.max(values) - np.min(values)) * 255
    image_height, image_width = 100, 100
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

    for (x, y), value in zip(points, values_nor):
        color = cv2.applyColorMap(np.uint8(values_nor), cv2.COLORMAP_VIRIDIS)
        color = color[0][0]

        cv2.drawMarker(image, (int(y), int(x)), (int(color[0]), int(color[1]), int(color[2])), 1, 2, 1)

    cv2.imwrite(img_name, image)
