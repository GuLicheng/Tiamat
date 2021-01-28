import torch
import torch.nn.functional as F

x = [
    [1, 0, 1, 0],
    [0, 2, 0, 2],
    [1, 1, 1, 1]
]

w_key = [
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 0]
]

w_query = [
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1]
]

w_value = [
    [0, 2, 0],
    [0, 3, 0],
    [1, 0, 3],
    [1, 1, 0]
]
x = torch.tensor(x, dtype=torch.float32)
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

keys = x @ w_key
queries = x @ w_query
values = x @ w_value

print(keys)
print(queries)
print(values)

attention_scores = queries @ keys.T
print(attention_scores)
attention_scores_softmax = F.softmax(attention_scores, dim=-1)
print(attention_scores_softmax)

weight_values = values[:, None] * attention_scores_softmax.T[:, :, None]
print(weight_values)