from preprocessing import read_tfrecords
import numpy as np

def calculate_cosine_similarity(x1, y1, x2, y2, direction_deg):
    # Calculate the vectors from the first datapoint to the second datapoint and the direction
    direction_deg = (180 - direction_deg) % 360
    vec_to_point = np.array([x2 - x1, y2 - y1])
    vec_direction = np.array([np.cos(np.radians(direction_deg)), np.sin(np.radians(direction_deg))])
    
    # Calculate the cosine similarity
    cosine_similarity = np.dot(vec_to_point, vec_direction) / (np.linalg.norm(vec_to_point) * np.linalg.norm(vec_direction))

    if (cosine_similarity <0):
        cosine_similarity = 0

    return cosine_similarity

def apply_cosine_similarity(firemask,direction_matrix2, func):
    '''Calculates Cosine Similarity for all fire pixels in firemask on the whole raster'''
    indices = np.argwhere(firemask == 1)

    res_list = list()

    for i, sublist in enumerate(indices):
        #x, y = np.meshgrid(np.arange(64), np.arange(64))
        x_f = sublist[0] # 0
        y_f = sublist[1] # 0
        wind_direction = direction_matrix2[x_f,y_f] # 45

        result = np.zeros((direction_matrix2.shape[0], direction_matrix2.shape[1]))

        for x in range(direction_matrix2.shape[0]):
            for y in range(direction_matrix2.shape[1]):
                res = func(x_f,y_f,x,y, wind_direction)
                result[x,y] = res
        result[np.isnan(result)] = 1
        res_list.append(result)

    sum_arrays = sum(res_list)
    min_val = np.min(sum_arrays)
    max_val = np.max(sum_arrays)

    normalized_array = (sum_arrays - min_val) / (max_val - min_val)
    
    return normalized_array


    def add_fire_direction_to_tensor(
        tensor: torch.Tensor,
        wind_direction_index: int,
        fire_mask_index: int) -> torch.Tensor:

    fire_mask = tensor[:, :, :, fire_mask_index].numpy()
    wind_direction = tensor[:, :, :, wind_direction_index].numpy()

    tensor_shape = (64, 64, 1)
    fire_direction_final = torch.empty((tensor.shape[0],) + tensor_shape)

    for i in range(tensor.shape[0]):
        wind_influence = apply_cosine_similarity(fire_mask[i,:,:], wind_direction[i,:,:],calculate_cosine_similarity)
        wind_influence = torch.from_numpy(wind_influence)
        wind_influence = wind_influence.unsqueeze(-1) # Unsqueeze adds dimension m to make it compatible
        fire_direction_final[i] = wind_influence

    tensor = torch.cat([fire_direction_final, tensor], dim=-1)

    return tensor