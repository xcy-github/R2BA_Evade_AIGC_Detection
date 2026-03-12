from io import BytesIO
import cv2
import os
import torch
import torch.nn
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import random
from alibabacloud_green20220302.client import Client
from alibabacloud_green20220302 import models
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.client import Client as UtilClient
from alibabacloud_tea_util import models as util_models
import json
import uuid
import oss2
import time
import os
import ast
# 服务是否部署在vpc上
is_vpc = False
# 文件上传token endpoint->token
token_dict = dict()
# 上传文件客户端
bucket = None


# 创建请求客户端
def create_client(access_key_id, access_key_secret, endpoint):
    config = Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        # 设置http代理。
        http_proxy='http://10.10.1171.10171:10171',
        # 设置https代理。
        # https_proxy='https://10.10.xx.xx:xxxx',
        # 接入区域和地址请根据实际情况修改。
        endpoint=endpoint
    )
    return Client(config)


# 创建文件上传客户端
def create_oss_bucket(is_vpc, upload_token):
    global token_dict
    global bucket
    auth = oss2.StsAuth(upload_token.access_key_id, upload_token.access_key_secret, upload_token.security_token)

    if (is_vpc):
        end_point = upload_token.oss_internal_end_point
    else:
        end_point = upload_token.oss_internet_end_point
    # 注意：此处实例化的bucket请尽可能重复使用，避免重复建立连接，提升检测性能。
    bucket = oss2.Bucket(auth, end_point, upload_token.bucket_name)


# 上传文件
def upload_file(file_name, upload_token):
    create_oss_bucket(is_vpc, upload_token)
    object_name = upload_token.file_name_prefix + str(uuid.uuid1()) + '.' + file_name.split('.')[-1]
    bucket.put_object_from_file(object_name, file_name)
    return object_name


def invoke_function(access_key_id, access_key_secret, endpoint, file_path):
    # 注意：此处实例化的client请尽可能重复使用，避免重复建立连接，提升检测性能。
    client = create_client(access_key_id, access_key_secret, endpoint)
    # 创建RuntimeObject实例并设置运行参数。
    runtime = util_models.RuntimeOptions()

    # 本地文件的完整路径，例如D:\localPath\exampleFile.png
    # file_path = '/home/DATASET_200/ImageNet/ldm_200/fake/auannvrmrm.png'

    # 获取文件上传token
    upload_token = token_dict.setdefault(endpoint, None)
    if (upload_token == None) or int(upload_token.expiration) <= int(time.time()):
        response = client.describe_upload_token()
        upload_token = response.body.data
        token_dict[endpoint] = upload_token
    # 上传文件
    object_name = upload_file(file_path, upload_token)

    # 检测参数构造。
    service_parameters = {
        # 待检测文件所在bucket名称。
        'ossBucketName': upload_token.bucket_name,
        # 待检测文件。
        'ossObjectName': object_name,
        # 数据唯一标识
        'dataId': str(uuid.uuid1())
    }

    image_moderation_request = models.ImageModerationRequest(
        # 图片检测service：内容安全控制台图片增强版规则配置的serviceCode，示例：baselineCheck
      	# 支持service请参考：https://help.aliyun.com/document_detail/467826.html?0#p-23b-o19-gff
        service='aigcDetector',
        service_parameters=json.dumps(service_parameters)
    )

    try:
        return client.image_moderation_with_options(image_moderation_request, runtime)
    except Exception as err:
        print(err)


def file_paths_function(img_dir):
    file_paths = [x.path for x in os.scandir(img_dir) if x.name.endswith(".png") or x.name.endswith(".jpg")]
    return file_paths


# 检测结果和置信度 result confidence
def Alibaba_API(img_path):  # 检测结果和置信度 result confidence
    access_key_id = ' '
    access_key_secret = ' '
    com_str_1 = 'green-cip.cn-shanghai.aliyuncs.com'
    response = invoke_function(access_key_id, access_key_secret, com_str_1, img_path)
    if response is not None:
        if UtilClient.equal_number(500, response.status_code) or \
                (response.body is not None and 200 != response.body.code):
            com_str = 'green-cip.cn-beijing.aliyuncs.com'
            response = invoke_function(access_key_id, access_key_secret, com_str, img_path)
        if response.status_code == 200:
            result = response.body
            if result.code == 200:
                result_data = result.data
                res_str = str(result_data.result[0])
                str_ls = ast.literal_eval(res_str)
                Confidence = str_ls['Confidence']
                Confidence_num = (float(Confidence)/100)/2  # 置信度
                if "aigc" in res_str:
                    res = 1  # 检测为假 1
                    fitness = 0.5 + Confidence_num
                else:
                    res = 0  # 检测为假 1
                    fitness = 0.5 - Confidence_num
                return fitness
    return "error"


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
def fitness_function(image, position, img_save_path):
    # 应用噪声处理
    processed_image = attack_process(image, position)
    processed_image_array = np.array(processed_image)
    processed_image_array = np.clip(processed_image_array, 0, 256).astype(np.uint8)
    processed_image = Image.fromarray(processed_image_array)
    processed_image.save(img_save_path)
    fitness = Alibaba_API(img_save_path)  # 0:UGC  1:AIGC
    ssim_this = calculate_ssim(np.array(image), np.array(processed_image))
    return fitness, processed_image, ssim_this


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
    particles = [Particle() for _ in range(num)]
    # 初始化粒子参数
    for particle in particles:
        fitness, processed_image, ssim_this = fitness_function(image, particle.position, output_path)
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
            fitness, processed_image, ssim_this = fitness_function(image, particle.position, output_path)
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
        if global_best_fitness < 0.5 and iterations % 2 == 0:  # 假设分类器输出小于0.5为停止条件
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


def pso_attack(file_paths, img_root_save, txt_path):
    res = [0, 0, 0]
    iter_use_sum = 0
    for ori_img_path in file_paths:
        img_name = os.path.basename(ori_img_path)
        adv_img_path = img_root_save + img_name
        ori_img = Image.open(ori_img_path)
        fitness = Alibaba_API(ori_img_path)  # 原始检测结果
        res[0] = res[0] + 1 if fitness > 0.5 else res[0]  # 检测为AIGC
        param_adv, y_adv, ssim_adv, iter_use = pso_ssim(ori_img, adv_img_path)
        res[1] = res[1] + 1 if fitness > 0.5 and y_adv > 0.5 else res[1]  # 原始AIGC 攻击AIGC 即失败
        res[2] = res[2] + ssim_adv if fitness > 0.5 else res[2]  # 攻击成功的图像质量
        iter_use_sum = iter_use_sum + iter_use if fitness > 0.5 else iter_use_sum
        with open(txt_path, "a") as f:
            print(f"{ori_img_path}: {fitness, y_adv, ssim_adv, iter_use*num}", file=f)
    res[2] = round(res[2] / res[0], 6)  # 平均SSIM数值
    res.append(round((res[0] - res[1]) / res[0], 6))  # 攻击成功率
    res.append(round(iter_use_sum * num / res[0], 3))  # 平均查询次数
    with open(txt_path, "a") as f:
        print(f"img_save: {img_root_save}", file=f)
        print(f"[ori:adv:ssim:sr:query]: {res}, total_num: {len(file_paths)}", file=f)


# 主程序
if __name__ == "__main__":
    gb_Ksize, gb_sigma_min, gb_sigma_max = 6, 0.5, 5
    jpeg_min, jpeg_max = 10, 100
    gn_sigma = (5 / 255) * (5 / 255)
    light_var, img_size, cir_min, cir_max = 0.8, 200, 30, 100
    max_w, min_w = 5, 1
    c1, c2, random_p = 1.5, 1.5, 0.5
    num = 10
    RESULT_PATH = "/ALL200_ALIBABA/Ours/"
    DATASET_PATH = [["/DATASET/ImageNet/ProGAN/", "ImageNet/ProGAN"],
                    ["/DATASET/GenImage/VQDM/", "GenImage/VQDM"],
                    ["/DATASET/ImageNet/ldm_200", "ImageNet/ldm_200"]]

    for dataset_ATTACK, save_ATTACK in DATASET_PATH[:1]:
        TXT_PATH = RESULT_PATH + f"{save_ATTACK}.txt"
        SAVE_ROOT_PATH = RESULT_PATH + f"{save_ATTACK}/"
        os.makedirs(SAVE_ROOT_PATH, exist_ok=True)
        with open(TXT_PATH, "a") as f_txt:
            print(dataset_ATTACK, SAVE_ROOT_PATH, file=f_txt)
            print(f"[{gb_Ksize}, {gb_sigma_min}, {gb_sigma_max}, {jpeg_min}, {jpeg_max}, {gn_sigma}, "
                  f"{light_var}, {img_size}, {cir_min}, {cir_max}, {max_w}, {min_w}, {c1}, {c2}, {random_p}]",
                  file=f_txt)
        file_paths = file_paths_function(dataset_ATTACK)
        pso_attack(file_paths, SAVE_ROOT_PATH, TXT_PATH)
