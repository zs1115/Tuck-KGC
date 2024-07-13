import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 tokenizer, pooler, use_rels, rel_policy, sample_paths,
                 trf_layers, trf_heads, trf_hidden, drop, use_distances, max_seq_len,
                 sample_rels, triples, ablate_anchors, device,
                 double_entity_embedding=False, double_relation_embedding=False, triple_relation_embedding=False, triple_entity_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.u = 1.0

        self.pooler = pooler
        self.use_rels = use_rels
        self.policy = rel_policy
        self.sample_paths = sample_paths
        self.use_distances = use_distances
        self.max_seq_len = max_seq_len
        self.sample_rels = sample_rels
        self.drop = drop
        self.triples = triples
        self.ablate_anchors = ablate_anchors
        self.device = device


        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.entity_dim = hidden_dim * 3 if triple_entity_embedding else self.entity_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        self.relation_dim = hidden_dim * 3 if triple_relation_embedding else self.relation_dim

        # anchors hashing mechanism

        self.set_enc = nn.Sequential(
            nn.Linear(self.entity_dim * (self.sample_paths + self.sample_rels), self.entity_dim * 2), nn.Dropout(drop), nn.ReLU(),
            nn.Linear(self.entity_dim * 2, self.entity_dim)
        ) if not self.ablate_anchors else nn.Sequential(
            nn.Linear(self.entity_dim * sample_rels, self.entity_dim * 2), nn.Dropout(drop), nn.ReLU(),
            nn.Linear(self.entity_dim * 2, self.entity_dim)
        )

        # init
        for module in self.set_enc.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.anchor_embeddings = nn.Embedding(num_embeddings=len(tokenizer.token2id)+1, embedding_dim=self.entity_dim)
        nn.init.uniform_(
            tensor=self.anchor_embeddings.weight,  # .weight for Embedding
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # back to normal relation embs, +1 for the padding relation
        self.relation_embedding = nn.Embedding(num_embeddings=nrelation, embedding_dim=self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
