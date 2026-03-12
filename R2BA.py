from ImageNet_Model import (
    device, get_FatFormer_model, get_ResNetND_model, get_UnivFD_model, get_DRCT_model, get_CNNSpot_model
)
from io import BytesIO
import cv2
import os
import torch
import torch.nn
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import random
from torch.utils.data import Dataset, DataLoader

cpu_num = 5  # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)


class CustomImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths  # 存储文件路径列表
        self.transform = transform  # 预处理方法

    def __len__(self):
        return len(self.file_paths)  # 返回数据集大小

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]  # 获取当前索引的图像路径
        image = Image.open(img_path).convert('RGB')  # 打开图像文件
        if self.transform:  # 如果有预处理方法，则应用
            image = self.transform(image)
        return image


def data_load_function(trans, img_dir):
    file_paths = [x.path for x in os.scandir(img_dir)]
    dataset = CustomImageDataset(file_paths=file_paths, transform=trans)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader, file_paths


def test_other_model(img_save_root_path, txt_path):
    for i_test_model in test_model2attack:
        if i_test_model == 0:
            test_model, test_trans, test_class = get_CNNSpot_model()
        elif i_test_model == 1:
            test_model, test_trans, test_class = get_ResNetND_model()
        elif i_test_model == 2:
            test_model, test_trans, test_class = get_UnivFD_model()
        elif i_test_model == 3:
            test_model, test_trans, test_class = get_FatFormer_model()
        elif i_test_model == 4 or i_test_model == 5:
            test_model, test_trans, test_class = get_DRCT_model(choose_ckpt, MODEL_CHOICE[i_test_model])
        else:
            test_model, test_trans, test_class = None, None, None
        ori_dataloader, _ = data_load_function(test_trans, dataset_ATTACK)
        adv_dataloader, _ = data_load_function(test_trans, img_save_root_path)
        res = [0, 0]
        for img, adv_img in zip(ori_dataloader, adv_dataloader):
            img = img.to(device)
            adv_img = adv_img.to(device)
            if test_class == 2:
                y_ori = test_model(img).softmax(dim=1)[:, -1:].item()
                y_adv = test_model(adv_img).softmax(dim=1)[:, -1:].item()
            else:
                y_ori = test_model(img).sigmoid().item()
                y_adv = test_model(adv_img).sigmoid().item()
            res[0] += 1 if y_ori > 0.5 else 0  # 识别错误的图像
            res[1] += 1 if y_ori > 0.5 and y_adv > 0.5 else 0  # 识别错误，但是攻击成功的数目
        res.append(round((res[0] - res[1]) / res[0], 6))
        with open(txt_path, "a") as print_res:
            print(f"Black == {MODEL_CHOICE[i_test_model]}[ori:adv:sr]: {res}, "
                  f"total_num: {len(ori_dataloader)}", file=print_res)


def file_paths_function(img_dir):
    file_paths = [x.path for x in os.scandir(img_dir) if x.name.endswith(".png") or x.name.endswith(".jpg")]

    return file_paths


# 模拟分类器
def simulated_classifier(image, classifier, trans, num_class):
    with torch.no_grad():
        img = trans(image).unsqueeze(0).to(device)
        if num_class == 2:
            prob = classifier(img).softmax(dim=1)[:, -1:].item()
        else:
            prob = classifier(img).sigmoid().item()
    return prob


def calculate_ssim(image1, image2):
    # image1 = image1.astype(np.float32)
    # image2 = image2.astype(np.float32)
    ssim_value = ssim(image1[:, :, 0], image2[:, :, 0], data_range=255) + \
                 ssim(image1[:, :, 1], image2[:, :, 1], data_range=255) + \
                 ssim(image1[:, :, 2], image2[:, :, 2], data_range=255)
    return ssim_value / 3


# 高斯模糊
def apply_gaussian_blur(image, ksize, sigmaX):
    image_array = np.array(image)
    blurred_array = cv2.GaussianBlur(image_array, (ksize, ksize), sigmaX)
    blurred_image = Image.fromarray(blurred_array)
    return blurred_image


# 定义图像压缩函数
def apply_compression(image, quality):
    # 将输入的图像转换为字节流
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=quality)
    compressed_image = Image.open(img_byte_arr)
    return compressed_image


# 定义高斯噪声函数
def apply_gaussian_noise(image, var, mean=0):
    img_array = np.array(image)
    sigma = var ** 0.5
    noise = np.random.normal(mean, sigma, img_array.shape)
    noisy_image = img_array + noise * 255  # 调整噪声幅度
    return Image.fromarray(np.clip(noisy_image, 0, 255).astype('uint8'))


# 定义亮度修改函数
def apply_brightness(image, brightness_factor, cx, cy, radius):
    img_array = np.array(image)
    height, width, _ = img_array.shape
    output_array = img_array.copy()
    # 计算距离的平方，以避免计算平方根
    for y in range(height):
        for x in range(width):
            distance_squared = (x - cx) ** 2 + (y - cy) ** 2
            if distance_squared < radius ** 2:
                relative_distance = np.sqrt(distance_squared) / radius
                current_brightness_factor = brightness_factor * (1 - relative_distance) + 1
                for c in range(3):  # 对每个通道进行处理
                    new_value = img_array[y, x, c] * current_brightness_factor
                    output_array[y, x, c] = np.clip(new_value, 0, 255)
    return Image.fromarray(output_array.astype('uint8'))


# 定义适应度函数
def fitness_function(image, position, classifier, trans, num_class):
    # 应用噪声处理
    processed_image = attack_process(image, position)
    processed_image_array = np.array(processed_image)
    processed_image_array = np.clip(processed_image_array, 0, 256).astype(np.uint8)
    processed_image = Image.fromarray(processed_image_array)
    classifier_output = simulated_classifier(processed_image, classifier, trans, num_class)
    ssim_this = calculate_ssim(np.array(image), np.array(processed_image))
    return classifier_output, processed_image, ssim_this


def attack_process(image, pos_input):
    processed_image = apply_gaussian_blur(image, pos_input[0], pos_input[1])
    processed_image = apply_compression(processed_image, pos_input[2])
    processed_image = apply_gaussian_noise(processed_image, var=pos_input[3])
    processed_image = apply_brightness(processed_image, pos_input[-1], pos_input[4], pos_input[5], pos_input[6])
    return processed_image


# 定义粒子类
class Particle:
    def __init__(self):
        # 初始化粒子的位置和速度
        self.position = [
            2 * random.randint(0, gb_Ksize) + 1, random.uniform(gb_sigma_min, gb_sigma_max),  # 高斯模糊半径与方差信息
            random.randint(jpeg_min, jpeg_max),  # 压缩质量
            random.uniform(0.0001, gn_sigma),  # 高斯噪声方差
            random.randint(30, img_size), random.randint(30, img_size),  # 亮斑调节
            random.randint(cir_min, cir_max), random.uniform(1 - light_var, 1 + light_var)
        ]
        self.best_position = self.position
        self.best_fitness = float('inf')
        self.best_ssim = float(0.0)
        self.image2save = None
        self.velocity = [random.uniform(-1, 1) for _ in range(len(self.position))]  # 初始化速度

    def update_velocity(self, global_best_position, w):
        # w = 2  # 惯性权重
        # c1 = 1.5  # 自我认知因子
        # c2 = 1.5  # 群体认知因子
        position_np = np.array(self.position)
        best_position_np = np.array(self.best_position)
        global_best_position_np = np.array(global_best_position)
        for i in range(len(self.velocity)):
            r1, r2 = random.random(), random.random()
            self.velocity[i] = (w * self.velocity[i] +
                                c1 * r1 * (best_position_np[i] - position_np[i]) +
                                c2 * r2 * (global_best_position_np[i] - position_np[i]))

    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            self.position[i] += np.random.choice([0, np.random.rand()], p=[random_p, 1 - random_p])

        self.position[0] = int(np.clip(self.position[0], 0, 2 * gb_Ksize + 1))
        self.position[0] = self.position[0] + 1 if self.position[0] % 2 == 0 else self.position[0]
        self.position[1] = np.clip(self.position[1], gb_sigma_min, gb_sigma_max)

        self.position[2] = int(np.clip(self.position[2], jpeg_min, jpeg_max))

        self.position[3] = np.clip(self.position[3], 0, gn_sigma)

        self.position[4] = int(np.clip(self.position[4], 30, img_size))
        self.position[5] = int(np.clip(self.position[5], 30, img_size))
        self.position[6] = int(np.clip(self.position[6], cir_min, cir_max))
        self.position[7] = np.clip(self.position[7], 1 - light_var, 1 + light_var)


# 定义粒子群优化函数  ssim
def pso_ssim(image, output_path):
    num = 100
    particles = [Particle() for _ in range(num)]
    # 初始化粒子参数
    for particle in particles:
        fitness, processed_image, ssim_this = fitness_function(image, particle.position, classifier, trans, num_class)
        particle.best_fitness = fitness
        particle.best_position = particle.position
        particle.best_ssim = ssim_this
        particle.image2save = processed_image
    global_best_position = None
    global_best_fitness = float('inf')
    iterations = 10  # 最大迭代次数
    for i in range(iterations):
        for particle in particles:
            # 计算适应度
            fitness, processed_image, ssim_this = fitness_function(image, particle.position, classifier, trans, num_class)
            # 更新粒子的最佳位置
            if particle.best_fitness < 0.5:  # 当攻击成功时，选择图像质量最优的粒子
                if ssim_this >= particle.best_ssim and fitness < 0.5:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position
                    particle.best_ssim = ssim_this
                    particle.image2save = processed_image
            else:  # 当攻击未成功时，选择概率最低的粒子
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position
                    particle.best_ssim = ssim_this
                    particle.image2save = processed_image
            # 更新全局最佳位置
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position
        # 根据最佳位置更新速度
        for particle in particles:
            particle.update_velocity(global_best_position, max_w - (max_w - min_w) * (i / iterations))
            particle.update_position()
        # 检查停止条件 并保存最佳SSIM数值的图像
        if global_best_fitness < 0.5:  # 假设分类器输出小于0.5为停止条件
            global_best_fitness_ls = [particle.best_fitness for particle in particles]
            filtered_indices = [i for i, value in enumerate(global_best_fitness_ls) if value < 0.5]
            ssim2save = 0
            best_image2save, fit2save, position2save = None, None, None
            for ind in filtered_indices:
                particle = particles[ind]
                ssim_ind = particle.best_ssim
                if ssim_ind >= ssim2save:
                    ssim2save = ssim_ind
                    best_image2save = particle.image2save
                    fit2save = particle.best_fitness
                    position2save = particle.best_position
            best_image2save.save(output_path)
            return position2save, fit2save, ssim2save, i+1
    # 攻击失败 则返回概率最低的图像
    global_best_fitness_ls = [particle.best_fitness for particle in particles]
    min_index = global_best_fitness_ls.index(min(global_best_fitness_ls))
    particle = particles[min_index]
    best_image2save = particle.image2save
    best_image2save.save(output_path)
    return particle.best_position, particle.best_fitness, particle.best_ssim, iterations


def pso_attack(file_paths, img_root_save, classifier, trans, num_class, txt_path, model_name, mode):
    if not test_only:
        res = [0, 0, 0]
        iter_use_sum = 0
        for ori_img_path in file_paths:
            img_name = os.path.basename(ori_img_path)
            adv_img_path = img_root_save + img_name
            ori_img = Image.open(ori_img_path)
            y_ori = simulated_classifier(ori_img, classifier, trans, num_class)
            res[0] = res[0] + 1 if y_ori > 0.5 else res[0]
            param_adv, y_adv, ssim_adv, iter_use = pso_ssim(ori_img, adv_img_path)
            res[1] = res[1] + 1 if y_ori > 0.5 and y_adv > 0.5 else res[1]
            res[2] = res[2] + ssim_adv if y_ori > 0.5 else res[2]
            iter_use_sum = iter_use_sum + iter_use
        print(res[0])
        res[2] = round(res[2] / res[0], 6)  # 平均SSIM数值
        res.append(round((res[0] - res[1]) / res[0], 6))  # 攻击成功率
        res.append(round(iter_use_sum * 100 / res[0], 3))  # 平均查询次数
        with open(txt_path, "a") as f:
            print(f"img_save: {img_root_save}", file=f)
            print(f"{mode}: White {model_name}[ori:adv:ssim:sr:query]: {res}, total_num: {len(file_paths)}", file=f)

    test_other_model(img_root_save, txt_path)


# 主程序
if __name__ == "__main__":
    gb_Ksize, gb_sigma_min, gb_sigma_max = 6, 0.5, 5
    jpeg_min, jpeg_max = 10, 100
    gn_sigma = (5 / 255) * (5 / 255)
    light_var, img_size, cir_min, cir_max = 0.8, 200, 30, 100
    max_w, min_w = 5, 1
    c1, c2, random_p = 1.5, 1.5, 0.5

    RESULT_PATH = "/"
    DATASET_PATH = [["/ImageNet/ProGAN/", "ImageNet/ProGAN"],
                    ["/GenImage/VQDM/", "GenImage/VQDM"],
                    ["/ImageNet/ldm_200", "ImageNet/ldm_200"]]
    MODEL_CHOICE = ["CNNSpot", "ResNetND", "UnivFD", "FatFormer", "UnivFD-Diff", "ConvB-Diff"]
    METHOD_CHOICE = ["PSO_SSIM"]

    choose_method = "PSO_SSIM"  # 选择攻击方法  0 PSO_SSIM
    choose_data = 2  # 选择攻击的数据集  0 ImageNet/ProGAN   1 GenImage/VQDM   2 ImageNet/ldm_200
    method_ATTACK = METHOD_CHOICE[choose_method]
    dataset_ATTACK, save_ATTACK = DATASET_PATH[choose_data][0], DATASET_PATH[choose_data][1]

    if choose_data == 0:  # 非扩散模型
        model2attack = [1]  # 想攻击的模型 0"CNNSpot" 1"ResNetND" 2"UnivFD" 3"FatFormer"
        test_only = True  # 默认为 False 即先生成再测试
        test_model2attack = [0, 1, 2, 3]  # 想测试的模型
        for i_model in model2attack:
            model_Flag = MODEL_CHOICE[i_model]
            TXT_PATH = RESULT_PATH + f"{save_ATTACK}/{method_ATTACK}/{method_ATTACK}_{model_Flag}.txt"
            SAVE_ROOT_PATH = RESULT_PATH + f"{save_ATTACK}/{method_ATTACK}/{model_Flag}/"
            os.makedirs(SAVE_ROOT_PATH, exist_ok=True)
            with open(TXT_PATH, "a") as f_txt:
                print(dataset_ATTACK, SAVE_ROOT_PATH, file=f_txt)
                print(f"[{gb_Ksize}, {gb_sigma_min}, {gb_sigma_max}, {jpeg_min}, {jpeg_max}, {gn_sigma}, "
                      f"{light_var}, {img_size}, {cir_min}, {cir_max}, {max_w}, {min_w}, {c1}, {c2}, {random_p}]",
                      file=f_txt)
            if i_model == 0:
                classifier, trans, num_class = get_CNNSpot_model()
            elif i_model == 1:
                classifier, trans, num_class = get_ResNetND_model()
            elif i_model == 2:
                classifier, trans, num_class = get_UnivFD_model()
            elif i_model == 3:
                classifier, trans, num_class = get_FatFormer_model()
            else:
                classifier, trans, num_class = None, None, None
                break
            file_paths = file_paths_function(dataset_ATTACK)
            pso_attack(file_paths, SAVE_ROOT_PATH, classifier, trans, num_class, TXT_PATH, model_Flag, method_ATTACK)
    else:  # 扩散模型
        model2attack = [4]  # 想攻击的模型  3"FatFormer" 4"UnivFD-Diff" 5"ConvB-Diff"
        test_only = False  # 默认为 False 即先生成再测试
        test_model2attack = [3, 4, 5]  # 想测试的模型
        choose_ckpt = "DRCT"
        for i_model in model2attack:
            model_Flag = MODEL_CHOICE[i_model]
            TXT_PATH = RESULT_PATH + f"{save_ATTACK}/{method_ATTACK}/{method_ATTACK}_{model_Flag}.txt"
            SAVE_ROOT_PATH = RESULT_PATH + f"{save_ATTACK}/{method_ATTACK}/{model_Flag}/"
            os.makedirs(SAVE_ROOT_PATH, exist_ok=True)
            with open(TXT_PATH, "a") as f_txt:
                print(dataset_ATTACK, SAVE_ROOT_PATH, file=f_txt)
                print(f"[{gb_Ksize}, {gb_sigma_min}, {gb_sigma_max}, {jpeg_min}, {jpeg_max}, {gn_sigma}, "
                      f"{light_var}, {img_size}, {cir_min}, {cir_max}, {max_w}, {min_w}, {c1}, {c2}, {random_p}]",
                      file=f_txt)
            if i_model == 3:
                classifier, trans, num_class = get_FatFormer_model()
            elif i_model == 4 or i_model == 5:
                classifier, trans, num_class = get_DRCT_model(choose_ckpt, model_Flag)
            else:
                classifier, trans, num_class = None, None, None
                break
            file_paths = file_paths_function(dataset_ATTACK)
            pso_attack(file_paths, SAVE_ROOT_PATH, classifier, trans, num_class, TXT_PATH, model_Flag, method_ATTACK)


