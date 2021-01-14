

from torch.distributions.beta import Beta
def get_lambda(self,
        batch_size):
    """
        Sample lambda given batch size.
    """
    dist = Beta(self.args.alpha, self.args.alpha)
    lambda_ = dist.sample(sample_shape=[bsz]).to("cuda")
    lambda_ = torch.max(lambda_, 1 - lambda_)
    return lambda_


def encoder_forward(self,
        lambda_,
        src_tokens_a,
        src_lengths_a,
        src_tokens_b,
        src_lengths_b,
        ...):
    """
        Args:
            lambda_ (FloatTensor): lambda used to permute sentences of shape `(batch)`
            src_tokens_a (LongTensor): tokens in the source sentence X of shape `(batch, src_len)`
            src_lengths_a (LongTensor): lengths of each source sentence X of shape `(batch)`
            src_tokens_b (LongTensor): tokens in the source sentence X' of shape `(batch, src_len)`
            src_lengths_b (LongTensor): lengths of each source sentence X' of shape `(batch)`
    """

        if self.layer_wise_attention:
            return_all_hiddens = True

        xa, encoder_embedding_a = self.forward_embedding(src_tokens_a)
        xb, encoder_embedding_b = self.forward_embedding(src_tokens_b)

        x = xa * lambda_.reshape(-1, 1, 1) + xb * (1-lambda_).reshape(-1, 1, 1)
        encoder_embedding = encoder_embedding_a * lambda_.reshape(-1, 1, 1) + \
                encoder_embedding_b * (1 - lambda_).reshape(-1, 1, 1)


        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens_a.eq(self.padding_idx)

        ......


def decoder_extract_features(self,
    lambda_,
    prev_output_tokens_a,
    perv_output_tokens_b,
    ...):

    """
        Args:
            lambda_ (FloatTensor): lambda used to permute sentences of shape `(batch)`
            prev_output_tokens_a (LongTensors): previous decoder outputs of target sentence Y of shape `(batch, tgt_len)`
            prev_output_tokens_b (LongTensors): previous decoder outputs of target sentence Y' of shape `(batch, tgt_len)`
    """

    def get_embedding(prev_output_tokens):
        ......

    xa = get_embedding(prev_output_tokens_a)
    xb = get_embedding(prev_output_tokens_b)

    x = xa * lambda_.view(-1, 1, 1) + xb * (1-lambda_).view(-1, 1, 1)

    ......


def labeled_smooth_cross_entropy_forward(self,
        model,
        sample,
        lambda_,
        reduce=True):

    net_output = model(lambda_,
                    sample["net_input_a"]["src_tokens"],
                    sample["net_input_a"]["src_lengths"],
                    sample["net_input_b"]["src_tokens"],
                    sample["net_input_b"]["src_lengths"],
                    sample["net_input_a"]["prev_output_tokens"],
                    prev_output_tokens_b=sample["net_input_b"]["prev_output_tokens"])
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    loss_a, nll_loss_a = label_smoothed_nll_loss(
            lprobs, sample["target_a"].view(-1,1), self.eps, ignore_index=self.padding_idx, reduce=False,
    )
    loss_b, nll_loss_b = label_smoothed_nll_loss(
            lprobs, sample["target_b"].view(-1,1), self.eps, ignore_index=self.padding_idx, reduce=False,
    )

    bsz, slen = sample["target_a"].size()
    loss_a = loss_a.reshape(bsz, slen)
    nll_loss_a = nll_loss_a.reshape(bsz, slen)
    loss_b = loss_b.reshape(bsz, slen)
    nll_loss_b = nll_loss_b.reshape(bsz, slen)

    loss = loss_a * lambda_.view(-1, 1) + loss_b * (1-lambda_).view(-1, 1)
    nll_loss = nll_loss_a * lambda_.view(-1, 1) + nll_loss_b * (1-lambda_).view(-1, 1)
    valid_indices = (sample["target_a"] != self.padding_idx)
    loss = loss * valid_indices.float()
    nll_loss = nll_loss * valid_indices.float()

    if reduce:
        loss = loss.sum()
        nll_loss = nll_loss.sum()

    ......


