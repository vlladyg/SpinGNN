import torch 


term_H = torch.eye(3)

term_DM1 = torch.tensor([[0., 0.,  0.], 
                         [0., 0.,  1.], 
                         [0., -1., 0.]])

term_DM2 = torch.tensor([[0., 0.,  -1.], 
                         [0., 0.,  0.], 
                         [1., 0., 0.]])

term_DM3 = torch.tensor([[0., 1.,  0.], 
                         [-1., 0.,  0.], 
                         [0., 0., 0.]])

term_ASEI1 = torch.tensor([[1., 0.,  0.], 
                         [0., -1.,  0.], 
                         [0., 0., 0.]])

term_ASEI2 = torch.tensor([[0., 1.,  0.], 
                         [1., 0.,  0.], 
                         [0., 0., 0.]])

term_ASEI3 = torch.tensor([[0., 0.,  1.], 
                         [0., 0.,  0.], 
                         [1., 0., 0.]])

term_ASEI4 = torch.tensor([[0., 0.,  0.], 
                         [0., 0.,  1.], 
                         [0., 1., 0.]])

term_ASEI5 = torch.tensor([[0., 0.,  0.], 
                          [0., -1.,  0.], 
                          [0., 0., 1.]])


matrix_terms_list_J = [term_H, term_DM1, term_DM2, term_DM3,
                             term_ASEI1, term_ASEI2, term_ASEI3, term_ASEI4, term_ASEI5]

matrix_terms_list_A = [term_H, term_ASEI1, term_ASEI2, term_ASEI3, term_ASEI4, term_ASEI5]

matrix_terms_J = torch.concat([el.unsqueeze(0) for el in matrix_terms_list_J])
matrix_terms_A = torch.concat([el.unsqueeze(0) for el in matrix_terms_list_A])