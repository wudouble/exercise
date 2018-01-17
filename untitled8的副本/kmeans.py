import numpy as np

def kmeans(dataset,k,maxT):
    points_num,featuer_num = dataset.shape
    data_matrix=np.zeros((points_num,featuer_num +1))
    data_matrix[:,:-1] = dataset

#   选中心点
    center_points = data_matrix[np.random.randint(points_num,size=k),:]
#   随机标记最后一列
    center_points[:,-1] = range(1,k+1)
#   初始化迭代次数
    iteration = 0
    old_center_points = None
    while not shouldstop(old_center_points,center_points,k,iteration):
        old_center_points = np.copy(center_points)
        iteration += 1
        update_dataset_label(data_matrix,center_points)
        center_points = update_center_points(data_matrix,center_points,k)

    return data_matrix



def shouldstop(old_center_points,center_points,k,iteration):
    if iteration > k:
        return True
    else:
        return np.array_equal(old_center_points,center_points)

def update_dataset_label(data_matrix, center_points):
    num_points = data_matrix.shape[0]
    for i in range(0,num_points):
        data_matrix[i,-1] = get_label_from_centerpoints(data_matrix[i,:-1], center_points)



def get_label_from_centerpoints(data_matrix_row, center_points):

    minDistance = np.linalg.norm(data_matrix_row - center_points[0,:-1])
    label = center_points[0,-1]
    for i in range(1,center_points.shape[0]):
        temp = np.linalg.norm((data_matrix_row - center_points[i,:-1]))
        if minDistance > temp:
            minDistance = temp
            label = center_points[i,-1]
    return label

def update_center_points(data_matrix,center_points,k):
    for i in range(0,k):
        data = data_matrix[data_matrix[:,-1] ==center_points[i,-1],:]
        center_points[i,:-1] = np.mean(data[:,-1])
       # center_points[i,-1] = center_points[i,-1]
    return center_points

x1 = np.array([1,1])
x2 = np.array([2,1])
x3 = np.array([4,3])
x4 = np.array([5,3])
x5 = np.array([8,2])
test_set = np.vstack((x1,x2,x3,x4,x5))
print(test_set)
result = kmeans(test_set,2,10)
print(result)













