from datetime import time
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from base.base_trainer import BaseTrainer
from utils.rank_metrics import calc_mAP, calc_metrics
from easydict import EasyDict as edict

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def unpack_batch_to_input_DME(batch, device, default=None):
    qttid = getattr(batch,"query_token_type_ids", default)
    if qttid is not None: qttid = qttid.to(device)
    returnDict = edict(
        query_enc_text_ids=None, 
        query_enc_text_attn_mask=batch.query_text_mask.to(device),
        query_enc_token_type_ids=qttid,
        query_enc_position_ids=None,
        query_enc_head_mask=None, 
        query_enc_inputs_embeds=batch.pad_query_text_feat.to(device),
        query_enc_vis_embeds=batch.pad_query_vis_feat.to(device), 
        query_enc_vis_attn_mask=batch.query_vis_mask.to(device), 
        query_enc_vis_token_type_ids=None,
        query_enc_image_text_alignment=batch.image_2_text_alignment.to(device),
        query_enc_output_attn=None,
        query_enc_output_hidden_states=None,

        ctx_enc_text_ids=None, 
        ctx_enc_text_attn_mask=batch.ctx_text_mask.to(device), 
        ctx_enc_token_type_ids=None, 
        ctx_enc_position_ids=None,
        ctx_enc_head_mask=None, 
        ctx_enc_inputs_embeds=batch.pad_ctx_text_feat.to(device),
        ctx_enc_vis_embeds=batch.pad_ctx_vis_feat.to(device), 
        ctx_enc_vis_attn_mask=batch.ctx_vis_mask.to(device), 
        ctx_enc_vis_token_type_ids=None,
        ctx_enc_image_text_alignment=None,
        ctx_enc_output_attn=None,
        ctx_enc_output_hidden_states=None,

        return_dict=True,
        return_embedding=True
    )
    return returnDict


def unpack_for_query(batch, device, default=None):
    qttid = getattr(batch,"query_token_type_ids",default)
    if qttid is not None: qttid = qttid.to(device)
    returnDict = edict(
        query_enc_text_ids=None, 
        query_enc_text_attn_mask=batch.query_text_mask.to(device),
        query_enc_token_type_ids=qttid,
        query_enc_position_ids=None,
        query_enc_head_mask=None, 
        query_enc_inputs_embeds=batch.pad_query_text_feat.to(device),
        query_enc_vis_embeds=batch.pad_query_vis_feat.to(device), 
        query_enc_vis_attn_mask=batch.query_vis_mask.to(device), 
        query_enc_vis_token_type_ids=None,
        query_enc_image_text_alignment=batch.image_2_text_alignment.to(device),
        query_enc_output_attn=None,
        query_enc_output_hidden_states=None,
        return_dict=True,
        return_embedding=True
    )
    return returnDict

def unpack_for_segment(batch, device, default=None):
    returnDict = edict(
        ctx_enc_text_ids=None, 
        ctx_enc_text_attn_mask=batch.ctx_text_mask.to(device), 
        ctx_enc_token_type_ids=None, 
        ctx_enc_position_ids=None,
        ctx_enc_head_mask=None, 
        ctx_enc_inputs_embeds=batch.pad_ctx_text_feat.to(device),
        ctx_enc_vis_embeds=batch.pad_ctx_vis_feat.to(device), 
        ctx_enc_vis_attn_mask=batch.ctx_vis_mask.to(device), 
        ctx_enc_vis_token_type_ids=None,
        ctx_enc_image_text_alignment=None,
        ctx_enc_output_attn=None,
        ctx_enc_output_hidden_states=None,
        return_dict=True,
        return_embedding=True
    )
    return returnDict

class Trainer_DME(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, val_seg_loader=None, test_data_loader=None, test_seg_loader=None,lr_scheduler=None, writer=None):
        super().__init__(model, loss, metrics, optimizer, config, writer)
        self.config = config
        self.data_loader = data_loader

        self.valid_data_loader = valid_data_loader
        self.valid_seg_loader = val_seg_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.test_seg_loader = test_seg_loader
        self.lr_scheduler = lr_scheduler
        self.batch_size = self.data_loader.batch_size
        self.best_mAP = 0.

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            if self.writer is not None:
                self.writer.log_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = [0] * len(self.data_loader)
        self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        with tqdm(self.data_loader, desc=f"Training epoch {epoch}", total=len(self.data_loader)) as progress:
            for batch_idx, batch_collect in enumerate(progress):
                # then assume we must tokenize the input, e.g. its a string
                self.optimizer.zero_grad()
                query_embeds, ctx_embeds = self.model(**unpack_batch_to_input_DME(batch_collect,device=self.device))
                output = sim_matrix(query_embeds, ctx_embeds)
                loss = self.loss(output)
                loss.backward()
                self.optimizer.step()

                detached_loss = loss.detach().item()

                if self.writer is not None:
                    iter_idx = batch_idx + epoch * len(self.data_loader)
                    self.writer.add_scalar('loss_train', detached_loss, iter_idx)

                total_loss[batch_idx] += detached_loss

                progress.set_postfix({"dl": batch_idx, "loss": detached_loss})

                self.optimizer.zero_grad()


        log = {
            f'loss_{batch_idx}': total_loss[batch_idx] for batch_idx in range(len(self.data_loader))
        }

        # TBD
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = [0] * len(self.valid_data_loader)
        # meta_arr = {x: [] for x in range(len(self.valid_data_loader))}
        query_embed_arr = []
        ctx_embed_arr = []

        query_ids = []
        ctx_ids = []
        gt = dict()

        with torch.no_grad():
            for batch_idx, batch_collect in enumerate(self.valid_data_loader):
       
                query_embeds, ctx_embeds = self.model(**unpack_batch_to_input_DME(batch_collect,device=self.device)) 

                # collect for eval
                query_embed_arr.append(query_embeds.cpu())
                for meta in batch_collect.meta:
                    query_ids.append(meta['query_id'])
                    gt.update({meta['query_id']:meta['answer_segment_id']})

                sims_batch = sim_matrix(query_embeds, ctx_embeds)
                loss = self.loss(sims_batch)
                total_val_loss[batch_idx] += loss.item()

        for batch_idx in range(len(self.valid_data_loader)):
            # TODO: this needs a clean
            if self.writer is not None:
                iter_idx = batch_idx + epoch * len(self.valid_data_loader)
                self.writer.add_scalar('loss_val', total_val_loss[batch_idx], iter_idx)
        
        res_dict = {
            f'val_loss_{batch_idx}': total_val_loss[batch_idx] for batch_idx in range(len(self.valid_data_loader))
        }

        if epoch%2==0:
            with torch.no_grad():
                for batch_idx, batch_collect in enumerate(self.valid_seg_loader):
                    if isinstance(self.model, nn.DataParallel):
                        ctx_embeds = self.model.module.compute_ctx(**unpack_for_segment(batch_collect,device=self.device))
                    else:
                        ctx_embeds = self.model.compute_ctx(**unpack_for_segment(batch_collect,device=self.device))
                    # collect for eval
                    ctx_embed_arr.append(ctx_embeds.cpu())
                    ctx_ids.extend(batch_collect['seg_id'])

            
            all_query = torch.cat(query_embed_arr, dim=0)
            all_ctx = torch.cat(ctx_embed_arr, dim=0)
            sims = sim_matrix(all_query, all_ctx).numpy()

            _, ranked_each_q, metrics_res = calc_metrics(
                ranking_scores=sims, query_id=query_ids, ctx_id=ctx_ids, gt=gt, higher_first=True,
                metrics=['mAP','r@1','r@5','r@10','r@50'], 
                cond_type2ids={
                    'all_query': query_ids,
                    'text_only': self.valid_data_loader.dataset.query_type['text'],
                    'video_only': self.valid_data_loader.dataset.query_type['video'],
                    't+v': self.valid_data_loader.dataset.query_type['text_video'],
                    'not_in_train': self.valid_data_loader.dataset.not_in_train,
                },
                sorted=False
            )

            for met in ['mAP','r@1','r@5','r@10','r@50']:
                self.writer.add_scalar(f'eval_{met}_all_query', metrics_res[met]['all_query'], epoch)
            
            if metrics_res['mAP']['all_query'] > self.best_mAP:
                self.best_mAP = metrics_res['mAP']['all_query']
                rt_test_dict = self._test_epoch(epoch)
            else:
                rt_test_dict = None
            
            self.save_metric['valid'].update(
                {epoch: metrics_res}
            )

            self.save_ranking['valid'].update(
                {epoch: ranked_each_q}
            )


            res_dict.update({
                f'val_{met}_{cond}': metrics_res[met][cond]  for met in metrics_res.keys() for cond in metrics_res[met].keys()
            })

            if rt_test_dict is not None:
                res_dict.update(rt_test_dict) 

        return res_dict

    def _test_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        query_embed_arr = []
        ctx_embed_arr = []

        query_ids = []
        ctx_ids = []
        gt = dict()

        with torch.no_grad():
            for batch_idx, batch_collect in enumerate(self.test_data_loader):
       
                query_embeds, _ = self.model(**unpack_batch_to_input_DME(batch_collect,device=self.device)) 

                # collect for eval
                query_embed_arr.append(query_embeds.cpu())
                for meta in batch_collect.meta:
                    query_ids.append(meta['query_id'])
                    gt.update({meta['query_id']:meta['answer_segment_id']})


        with torch.no_grad():
            for batch_idx, batch_collect in enumerate(self.test_seg_loader):
                if isinstance(self.model, nn.DataParallel):
                    ctx_embeds = self.model.module.compute_ctx(**unpack_for_segment(batch_collect,device=self.device))
                else:
                    ctx_embeds = self.model.compute_ctx(**unpack_for_segment(batch_collect,device=self.device))
                # collect for eval
                ctx_embed_arr.append(ctx_embeds.cpu())
                ctx_ids.extend(batch_collect['seg_id'])

        
        all_query = torch.cat(query_embed_arr, dim=0)
        all_ctx = torch.cat(ctx_embed_arr, dim=0)
        sims = sim_matrix(all_query, all_ctx).numpy()

        _, ranked_each_q, metrics_res = calc_metrics(
            ranking_scores=sims, query_id=query_ids, ctx_id=ctx_ids, gt=gt, higher_first=True,
            metrics=['mAP','r@1','r@5','r@10','r@50'], 
            cond_type2ids={
                'all_query': query_ids,
                'text_only': self.test_data_loader.dataset.query_type['text'],
                'video_only': self.test_data_loader.dataset.query_type['video'],
                't+v': self.test_data_loader.dataset.query_type['text_video'],
                'not_in_train': self.test_data_loader.dataset.not_in_train,
            },
            sorted=False
        )
        
        self.save_metric['test'].update(
            {epoch: metrics_res}
        )

        self.save_ranking['test'].update(
            {epoch: ranked_each_q}
        )
        
        res_dict = {}
        res_dict.update({
            f'test_{met}_{cond}': metrics_res[met][cond]  for met in metrics_res.keys() for cond in metrics_res[met].keys()
        })
        return res_dict

    def _progress(self, batch_idx, dl_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader[dl_idx], 'n_samples'):
            current = batch_idx * self.data_loader[dl_idx].batch_size
            total = self.data_loader[dl_idx].n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


if __name__ == "__main__":
    pass