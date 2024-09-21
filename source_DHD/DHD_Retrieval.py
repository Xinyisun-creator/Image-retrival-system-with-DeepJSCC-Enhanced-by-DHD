from .Dataloader import *
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def Evaluate_mAP(device, gallery_codes, query_codes, gallery_labels, query_labels, Top_N=None):
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        retrieval = (query_labels[i, :] @ gallery_labels.t() > 0).float()
        hamming_dist = (gallery_codes.shape[1] - query_codes[i, :] @ gallery_codes.t())

        retrieval = retrieval[T.argsort(hamming_dist)][:Top_N]
        retrieval_cnt = retrieval.sum().int().item()

        if retrieval_cnt == 0:
            continue

        score = T.linspace(1, retrieval_cnt, retrieval_cnt).to(device)
        index = (T.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float().to(device)

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP

def DoRetrieval(device, net, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)

    Crop_Normalize = T.nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in enumerate(Gallery_loader, 0):
            gallery_x_batch, gallery_y_batch = data[0].to(device), data[1].to(device)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in enumerate(Query_loader, 0):
            query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return mAP

def DoRetrieval_forPR(device, net, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS)
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS)
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=args.DHD_batch_size, num_workers=args.DHD_num_workers)

    Crop_Normalize = T.nn.Sequential(
        Kg.CenterCrop(224),
        Kg.Normalize(mean=T.as_tensor([0.485, 0.456, 0.406]), std=T.as_tensor([0.229, 0.224, 0.225]))
    )

    with T.no_grad():
        for i, data in enumerate(Gallery_loader, 0):
            gallery_x_batch, gallery_y_batch = data[0].to(device), data[1].to(device)
            gallery_x_batch = Crop_Normalize(gallery_x_batch)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)

        for i, data in enumerate(Query_loader, 0):
            query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
            query_x_batch = Crop_Normalize(query_x_batch)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)

    gallery_c = T.sign(gallery_c)
    query_c = T.sign(query_c)

    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    return gallery_c, query_c, gallery_y, query_y

def Compute_Precision_Recall(device, gallery_codes, query_codes, gallery_labels, query_labels, Top_N=None):
    num_query = query_labels.shape[0]
    precision_list = []
    recall_list = []
    
    # 定义固定的 Recall 取值范围
    recall_points = np.linspace(0, 1, 101)  # 从0到1，取101个点

    for i in range(num_query):
        retrieval = (query_labels[i, :] @ gallery_labels.t() > 0).float()
        hamming_dist = (gallery_codes.shape[1] - query_codes[i, :] @ gallery_codes.t())

        # 对 hamming_dist 和 retrieval 同时进行排序和截断
        sorted_indices = T.argsort(hamming_dist)
        retrieval = retrieval[sorted_indices][:Top_N]
        hamming_dist = hamming_dist[sorted_indices][:Top_N]

        retrieval_cnt = retrieval.sum().int().item()

        if retrieval_cnt == 0:
            continue

        # Calculate Precision and Recall
        precision, recall, _ = precision_recall_curve(retrieval.cpu().numpy(), -hamming_dist.cpu().numpy())

        # 插值到固定的 Recall 点
        if len(recall) < 2:
            continue  # 忽略无法计算的情况
        interp_precision = np.interp(recall_points, recall[::-1], precision[::-1])  # 需要反转数组
        precision_list.append(interp_precision)
        recall_list.append(recall_points)

    # 将 precision_list 转换为 numpy 数组
    precision_array = np.array(precision_list)

    # 计算 avg_precision 为每个 Recall 点上的平均 Precision
    avg_precision = np.mean(precision_array, axis=0)
    avg_recall = recall_points  # Recall 点是固定的

    return avg_precision, avg_recall

def Plot_Precision_Recall(avg_precision, avg_recall):
    plt.figure()
    plt.plot(avg_recall, avg_precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.grid(True)
    plt.savefig('Precision_Recall.png')
    plt.show()

# Example of how to use this function with your DoRetrieval function
def DoRetrievalWithPRCurve(device, net, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args):
    # Existing DoRetrieval code to get gallery_c, query_c, gallery_y, query_y
    gallery_c, query_c, gallery_y, query_y = DoRetrieval_forPR(device, net, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, args)
    
    # Calculate mAP
    mAP = Evaluate_mAP(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    
    # Compute and plot Precision-Recall curve
    avg_precision, avg_recall = Compute_Precision_Recall(device, gallery_c, query_c, gallery_y, query_y, Top_N)
    Plot_Precision_Recall(avg_precision, avg_recall)
    
    return mAP