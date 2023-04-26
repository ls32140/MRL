import torch
import torch.nn as nn

class cross_MoCo(nn.Module):
    def __init__(self, parameters, k_img_encoder, k_text_encoder, dim=512, K=42941, m=0.999, T=0.07, img_dim=4096, text_dim=300):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(cross_MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.k_img = k_img_encoder(img_dim, dim).cuda()
        self.k_text = k_text_encoder(text_dim, dim).cuda()

        self.parameters_pairs = [
            [parameters[0], self.k_img.parameters()],
            [parameters[1], self.k_text.parameters()]
        ]

        for parameters_pair in self.parameters_pairs:
            for param_q, param_k in zip(parameters_pair[0], parameters_pair[1]):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue_img = nn.functional.normalize(self.queue, dim=0).cuda()
        self.queue_text = nn.functional.normalize(self.queue, dim=0).cuda()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
            Momentum update of the key encoder
        """
        for parameters_pair in self.parameters_pairs:
            for param_q, param_k in zip(parameters_pair[0], parameters_pair[1]):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, str):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if (str == "img"):
            self.queue_img[:, ptr:ptr + batch_size] = keys.T
        if (str == "text"):
            self.queue_text[:, ptr: ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q_en, text_q_en ,im_k, text_k):
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            img_k_en = self.k_img(im_k)  # keys: NxC
            img_k_en = nn.functional.normalize(img_k_en, dim=1)

            text_k_en = self.k_text(text_k)  # keys: NxC
            text_k_en = nn.functional.normalize(text_k_en, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum("nc,nc->n", [im_q_en, text_k_en]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [im_q_en, self.queue_text.clone().detach().cuda()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= self.T
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            l_pos1 = torch.einsum("nc,nc->n", [text_q_en, img_k_en]).unsqueeze(-1)
            l_neg1 = torch.einsum("nc,ck->nk", [text_q_en, self.queue_img.clone().detach().cuda()])
            logits1 = torch.cat([l_pos1, l_neg1], dim=1)
            logits1 /= self.T
            labels1 = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            self._dequeue_and_enqueue(img_k_en, 'img')
            self._dequeue_and_enqueue(text_k_en, 'text')

            return logits, labels, logits1, labels1