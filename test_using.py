import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import warnings
import pandas as pd  # 用于生成表格
import datetime      # 用于获取当前时间作为文件名

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 内存友好型数据集定义 ---
class OffsetDataset(Dataset):
    """
    包装原始数据集，动态计算大类和小类标签，不占用额外内存。
    """
    def __init__(self, dataset, big_label, sub_offset):
        self.dataset = dataset
        self.big_label = big_label
        self.sub_offset = sub_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        # 动态生成复合标签 (大类, 小类)
        return img, (self.big_label, label + self.sub_offset)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature11 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5, inplace=False),
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5, inplace=False)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(39168, 128 * 8),
            nn.ReLU(),
            nn.Linear(128 * 8, 16 * 8),
            nn.ReLU(),
            nn.Linear(16 * 8, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, images, templates):
        x11 = self.feature11(images).cuda()
        x11 = self.conv11(x11)
        x11 = self.flatten(x11)
        x21 = templates.cuda()
        x21 = self.flatten(x21)
        x = torch.cat([x11, x21], dim=1)
        y = self.classifier(x)
        return y

def generate_data_small(images, templates, images_len, templates_len, images_labels, templates_labels, replication_factor):
    labels = torch.zeros(images_len * templates_len).cuda()
    images_final = images.repeat_interleave(templates_len, dim=0)
    labels_final = images_labels.repeat_interleave(templates_len, dim=0)
    templates_final = templates.repeat(images_len, 1, 1, 1)
    templates_labels_final = templates_labels.repeat(images_len)
    match_indices = (labels_final == templates_labels_final)
    labels[match_indices] = 1
    if replication_factor > 0:
        images_to_add = images_final[match_indices].repeat(replication_factor, 1, 1, 1)
        templates_to_add = templates_final[match_indices].repeat(replication_factor, 1, 1, 1)
        labels_to_add = labels[match_indices].repeat(replication_factor)
        images_final = torch.cat([images_final, images_to_add], dim=0)
        templates_final = torch.cat([templates_final, templates_to_add], dim=0)
        labels = torch.cat([labels, labels_to_add], dim=0)
    return images_final, templates_final, labels

# --- 2. 优化后的测试函数 ---
def test_hierarchical(big_model, small_model, test_loader, big_templates, small_templates):
    big_model.eval()
    small_model.eval()

    big_correct, small_correct = 0, 0
    test_total = len(test_loader.dataset)
    true_big_all, pred_big_all = [], []
    true_small_all, pred_small_all = [], []

    small_class_mapping = {0: (0, 10), 1: (10, 36), 2: (36, 46), 3: (46, 56)}

    with torch.no_grad():
        for test_images, (true_big, true_small) in tqdm(test_loader, desc="正在全量测试"):
            test_images, true_big = test_images.to(device), true_big.to(device)
            img_len = len(test_images)

            # 阶段一：大类预测
            b_temp_len = len(big_templates)
            b_temp_labels = torch.arange(0, b_temp_len).cuda()

            # 注意：此处 Batch Size 较小时不容易爆显存
            img_b, temp_b, _ = generate_data_small(test_images, big_templates, img_len, b_temp_len, true_big, b_temp_labels, b_temp_len-2)
            out_big = big_model(img_b, temp_b)
            vals_b, pred_b_bin = torch.max(out_big.data, 1)

            pred_big_batch = []
            cur = -1
            for i in range(img_len):
                score = [0] * b_temp_len
                for k in range(b_temp_len):
                    cur += 1
                    if pred_b_bin[cur] == 1: score[k] += vals_b[cur]
                pred_label = score.index(max(score))
                pred_big_batch.append(pred_label)

            # 统计大类结果
            true_big_all.extend(true_big.cpu().numpy())
            pred_big_all.extend(pred_big_batch)
            big_correct += (torch.tensor(pred_big_batch).cuda() == true_big).sum().item()

            # 阶段二：逐图进行小类预测
            for i in range(img_len):
                p_big = pred_big_batch[i]
                start, end = small_class_mapping[p_big]
                rel_temps = small_templates[start:end]
                n_small = len(rel_temps)

                # 显式确保标签在 GPU 上
                current_true_small = true_small[i:i + 1].to(device)
                current_small_temp_labels = torch.arange(0, n_small).to(device)

                img_s, temp_s, _ = generate_data_small(
                    test_images[i:i + 1],
                    rel_temps,
                    1,
                    n_small,
                    current_true_small,  # 使用已移至 GPU 的标签
                    current_small_temp_labels,  # 使用已移至 GPU 的模板标签
                    n_small - 2
                )
                out_small = small_model(img_s, temp_s)
                vals_s, pred_s_bin = torch.max(out_small.data, 1)

                score_s = [0] * n_small
                for k in range(n_small):
                    if pred_s_bin[k] == 1: score_s[k] += vals_s[k]

                p_small = score_s.index(max(score_s)) + start
                true_small_all.append(true_small[i].item())
                pred_small_all.append(p_small)
                if p_small == true_small[i].item(): small_correct += 1

            # 关键：显存清理
            torch.cuda.empty_cache()

        # --- 打印报表 ---
        print(f"\n大类总准确率: {100 * big_correct / test_total:.2f}%")
        cm_b = confusion_matrix(true_big_all, pred_big_all)
        names = ["image10", "EMNIST", "MNIST", "newimage10"]
        for i in range(len(cm_b)):
            print(f"大类 {i} ({names[i]}) 准确率: {100 * cm_b[i][i] / cm_b[i].sum():.2f}% (样本:{cm_b[i].sum()})")

        # ================= 新增：每个大类下的“小类整体准确率” =================
        print("\n--- 各大类内部的【小类整体准确率】 ---")
        true_small_arr = np.array(true_small_all)
        pred_small_arr = np.array(pred_small_all)

        for big_idx, (start, end) in small_class_mapping.items():
            # 利用 true_small_arr 提取出属于当前大类（比如 10 到 36）的所有样本
            mask = (true_small_arr >= start) & (true_small_arr < end)
            total_in_big = np.sum(mask)

            if total_in_big > 0:
                correct_in_big = np.sum(true_small_arr[mask] == pred_small_arr[mask])
                acc = 100 * correct_in_big / total_in_big
                print(
                    f"大类 {big_idx} ({names[big_idx]}) 内部小类准确率: {acc:.2f}% (正确/总数: {correct_in_big}/{total_in_big})")
            else:
                print(f"大类 {big_idx} ({names[big_idx]}): 无测试数据")

        print(f"\n小类总准确率: {100 * small_correct / test_total:.2f}%")
        cm_s = confusion_matrix(true_small_all, pred_small_all)
        all_indices = sorted(list(set(true_small_all)))
        print(f"{'小类索引':<8} | {'准确率':<10} | {'正确/总数'}")
        for idx in all_indices:
            r_idx = np.where(np.unique(true_small_all) == idx)[0][0]
            c_tot, c_cor = cm_s[r_idx].sum(), cm_s[r_idx][r_idx]
            print(f"{idx:<8} | {100 * c_cor / c_tot:>8.2f}% | {c_cor}/{c_tot}")

        # ================= 自动保存结果为表格和日志 =================
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存小类详细准确率到 CSV 表格
        small_stats_data = []
        for idx in all_indices:
            r_idx = np.where(np.unique(true_small_all) == idx)[0][0]
            c_tot = cm_s[r_idx].sum()
            c_cor = cm_s[r_idx][r_idx]
            acc = 100 * c_cor / c_tot if c_tot > 0 else 0
            small_stats_data.append({
                '小类标签': idx,
                '准确率(%)': round(acc, 2),
                '正确识别数': c_cor,
                '测试总数': c_tot
            })

        df_small = pd.DataFrame(small_stats_data)
        csv_filename = f'测试结果_小类详细_{timestamp}.csv'
        df_small.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n[文件已保存] 小类详细准确率表格已保存为: {csv_filename}")

        # 2. 保存大类结果和整体汇总到文本日志文件
        log_filename = f'测试日志_汇总报告_{timestamp}.txt'
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(f"测试时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 40 + "\n")
            f.write(f"测试集总样本量: {test_total}\n")
            f.write(f"大类分类整体准确率: {100 * big_correct / test_total:.4f}%\n")
            f.write(f"小类分类整体准确率: {100 * small_correct / test_total:.4f}%\n")
            f.write("=" * 40 + "\n")

            f.write("\n各大类识别准确率:\n")
            for i in range(len(cm_b)):
                acc = 100 * cm_b[i][i] / cm_b[i].sum() if cm_b[i].sum() > 0 else 0
                f.write(f"- 类别 {i} ({names[i]}): {acc:.2f}% (样本数: {cm_b[i].sum()})\n")

            f.write("\n各大类内部的【小类整体准确率】:\n")
            for big_idx, (start, end) in small_class_mapping.items():
                mask = (true_small_arr >= start) & (true_small_arr < end)
                total_in_big = np.sum(mask)
                if total_in_big > 0:
                    correct_in_big = np.sum(true_small_arr[mask] == pred_small_arr[mask])
                    acc = 100 * correct_in_big / total_in_big
                    f.write(
                        f"- 大类 {big_idx} ({names[big_idx]}): {acc:.2f}% (正确/总数: {correct_in_big}/{total_in_big})\n")

        print(f"[文件已保存] 总体测试日志已保存为: {log_filename}")

if __name__ == "__main__":
    tf = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Grayscale(1)])

    DATA_ROOT = "./datasets"

    # 1. 直接加载原数据集
    raw1 = datasets.ImageFolder(os.path.join(DATA_ROOT, "image10"),
                                transform=transforms.Compose([tf, transforms.Normalize([0.4334], [0.2070])]))
    raw2 = datasets.ImageFolder(os.path.join(DATA_ROOT, "EMNIST"),
                                transform=transforms.Compose([tf, transforms.Normalize([0.1793], [0.3288])]))
    raw3 = datasets.ImageFolder(os.path.join(DATA_ROOT, "MNIST"),
                                transform=transforms.Compose([tf, transforms.Normalize([0.0875], [0.2418])]))
    raw4 = datasets.ImageFolder(os.path.join(DATA_ROOT, "newimage10"),
                                transform=transforms.Compose([tf, transforms.Normalize([0.4725], [0.2366])]))

    # 2. 使用封装类实现流式加载（不占内存）
    ds1, ds2 = OffsetDataset(raw1, 0, 0), OffsetDataset(raw2, 1, 10)
    ds3, ds4 = OffsetDataset(raw3, 2, 36), OffsetDataset(raw4, 3, 46)

    # 3. 合并全量数据并创建加载器
    full_data = ConcatDataset([ds1, ds2, ds3, ds4])
    # 注意：batch_size 调小可以显著降低显存压力
    loader = DataLoader(full_data, batch_size=4, shuffle=False)

    # 4. 加载模型与模板 (此处路径请根据实际情况保持原样)
    big_model = Model().to(device)
    big_model.load_state_dict(torch.load('./llml20251204big_fullData_MainModel.pth', map_location=device)['model'])
    big_temps = torch.load("./detection/templates1/fulldata_bigClass_Knowledge.pth", map_location=device)

    small_model = Model().to(device)
    small_model.load_state_dict(torch.load('./llml20251201small_fullData_and_fullKnowledge_MainModel.pth', map_location=device)['model'])
    small_temps = torch.load('./detection/templates1/fulldata_smallClass_Knowledge.pth', map_location=device)

    test_hierarchical(big_model, small_model, loader, big_temps, small_temps)

"""
Created on 2026/3/25 9:09

@author: Administrator
"""
