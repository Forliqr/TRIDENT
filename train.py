import os
import gc
import sys
import random
import torchaudio
import torch
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import random
import numpy
from torch.nn import functional as F
from torchaudio.functional import resample
import numpy as np
from speechbrain.utils.parameter_transfer import Pretrainer
import traceback
import warnings
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
print(torch.cuda.device_count())

def safe_cosine_similarity(a, b, eps=1e-8):
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    a_norm = torch.norm(a, p=2, dim=-1, keepdim=True)
    b_norm = torch.norm(b, p=2, dim=-1, keepdim=True)
    zero_mask = (a_norm == 0) | (b_norm == 0)
    a_norm = a_norm + eps
    b_norm = b_norm + eps
    similarity = (a * b).sum(dim=-1) / (a_norm * b_norm).squeeze(-1)
    similarity = torch.where(zero_mask.squeeze(-1), torch.zeros_like(similarity), similarity)
    return similarity.clamp(-1.0, 1.0)

class SpeakerBrain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        wavs, lens1 = batch.sig
        tar_wavs, lens2 = batch.tar_sig
        flag = 0
    
        if not torch.isfinite(wavs).all() or not torch.isfinite(tar_wavs).all():
            flag = 1
            return torch.zeros(1, device=self.device, requires_grad=True), torch.zeros(1, device=self.device, requires_grad=True), torch.zeros(1, device=self.device, requires_grad=True), lens1, flag
                
        if stage == sb.Stage.TRAIN:
            wavs_aug_tot = []
            tar_wavs_aug_tot = []

            wavs_aug_tot.append(wavs)
            tar_wavs_aug_tot.append(tar_wavs)

            for count, augment in enumerate(self.hparams.augment_pipeline):
                wavs_aug = augment(wavs, lens1)
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs
            

            for count, augment in enumerate(self.hparams.augment_pipeline):
                wavs_aug = augment(tar_wavs, lens2)
                if wavs_aug.shape[1] > tar_wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : tar_wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(tar_wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    tar_wavs_aug_tot.append(wavs_aug)

                else:
                    tar_wavs = wavs_aug
                    tar_wavs_aug_tot[0] = tar_wavs 

            wavs = torch.cat(wavs_aug_tot, dim=0)
            tar_wavs = torch.cat(tar_wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens1 = torch.cat([lens1] * self.n_augment)
            lens2 = torch.cat([lens2] * self.n_augment)
        lens = torch.cat((lens1, lens2), dim=0)
        mix = torch.cat((wavs, tar_wavs), dim=0)

        feats = self.modules.compute_features(mix)
        feats = self.modules.mean_var_norm(feats, lens)
        
        source = feats[:feats.size(0)//2].unsqueeze(1)
        target = feats[feats.size(0)//2:].unsqueeze(1)
        feats = torch.cat([source, target], dim=1)

        embeddings = self.modules.embedding_model(feats)
        Method_embeddings, un_embeddings = self.modules.Method_Projection(embeddings)
        Target_embeddings = self.modules.Target_Projection(embeddings)

        embeddings = self.modules.Source_Projection(embeddings, un_embeddings)

        Target_outputs = self.modules.Target_classifier(Target_embeddings)
        Method_outputs = self.modules.Method_classifier(Method_embeddings)
        outputs = self.modules.Source_classifier(embeddings)

        return Method_outputs, Target_outputs, outputs, lens1, flag

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        Method_predictions, Target_predictions, Source_predictions, lens, flag = predictions
        if flag == 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        uttid = batch.id
        Sourceid, _ = batch.spk_id_encoded
        Targetid, _ = batch.Target_id
        Methodid, _ = batch.Method_id

        if stage == sb.Stage.TRAIN:
            Sourceid = torch.cat([Sourceid] * self.n_augment, dim=0)
            Targetid = torch.cat([Targetid] * self.n_augment, dim=0)
            Methodid = torch.cat([Methodid] * self.n_augment, dim=0)
        

        loss1 = self.hparams.compute_cost(Method_predictions, Methodid, lens)
        loss2 = self.hparams.compute_cost(Target_predictions, Targetid, lens)
        loss3 = self.hparams.compute_cost(Source_predictions, Sourceid, lens)
        samples_sign = torch.sign(torch.abs(Sourceid-Targetid)).flatten()
        selected_indices = torch.where(samples_sign==1)[0]
        selected_d = Source_predictions[selected_indices]
        selected_n = Target_predictions[selected_indices]
        loss4 = safe_cosine_similarity(selected_d, selected_n, eps=1e-6)
        loss4 = loss4.mean()
        _selected_indices = torch.where(samples_sign==0)[0]
        _selected_d = Source_predictions[_selected_indices]
        _selected_n = Target_predictions[_selected_indices]
        loss5 = safe_cosine_similarity(_selected_d, _selected_n, eps=1e-6)
        loss5 = loss5.mean()
        loss = 0.1 * loss1 + 0.1 * loss2 + loss3
        if not (torch.isnan(loss4).any() or torch.isinf(loss4).any() or (not torch.isfinite(loss4))):
            loss = loss + 0.1 * loss4
        if not (torch.isnan(loss5).any() or torch.isinf(loss5).any() or (not torch.isfinite(loss5))):
            loss = loss - 0.1 * loss5
        print(f'Loss:{loss} loss1:{loss1}   loss2:{loss2}   loss3:{loss3}   loss4:{loss4}   loss5:{loss5}')
        if torch.isnan(loss).any() or torch.isinf(loss).any() or (not torch.isfinite(loss)):
            loss = torch.clamp(loss, min=-1.0, max=30.0)
            self.optimizer.zero_grad()
            for param in self.modules.parameters():
                if param.numel() > 0:
                    if param.grad is not None:
                        param.grad.zero_()   
            return torch.tensor(0.0, device=self.device, requires_grad=True)   

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, Source_predictions, Sourceid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_checkpoint(
                meta={"ErrorRate": stage_stats["ErrorRate"]}
            )

def dataio_prep(hparams):
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    @sb.utils.data_pipeline.takes("wav", "lens")
    @sb.utils.data_pipeline.provides("sig")

    def audio_pipeline(wav, lens):

        num_frames = int(hparams["sample_rate"]*hparams["sentence_len"])
        sig, fs = torchaudio.load(
            wav
        )
        sig = resample(sig, fs, int(hparams["sample_rate"]))
        if sig.size(-1) < num_frames:

            sig = F.pad(sig, (0, num_frames-sig.size(-1)), "constant", 0)
        else:
            start = random.randint(0, sig.size(-1)-num_frames)
            sig = sig[:, start:start+num_frames]
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)


    @sb.utils.data_pipeline.takes("tar_wav")
    @sb.utils.data_pipeline.provides("tar_sig")

    def audio2_pipeline(tar_wav):

        num_frames = int(hparams["sample_rate"]*hparams["sentence_len"])
        if tar_wav != 'none':
            sig, fs = torchaudio.load(
                tar_wav
            )
            sig = resample(sig, fs, int(hparams["sample_rate"]))
            if sig.size(-1) < num_frames:

                sig = F.pad(sig, (0, num_frames-sig.size(-1)), "constant", 0)
            else:
                start = random.randint(0, sig.size(-1)-num_frames)
                sig = sig[:, start:start+num_frames]
            tar_sig = sig.transpose(0, 1).squeeze(1)
        else:
            tar_sig = torch.zeros(num_frames)

        return tar_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio2_pipeline)



    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        yield spk_id
        spk_id_encoded = label_encoder.encode_sequence_torch([spk_id])
        yield spk_id_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    @sb.utils.data_pipeline.takes("spk_id", "tar_wav", "id")
    @sb.utils.data_pipeline.provides("Target_name", "Target_id", "Method_id")
    def target_method_label_pipeline(spk_id, tar_wav, ID):
        if tar_wav == "none":
            Target_name = "none"
            yield Target_name
            Target_id = label_encoder.encode_sequence_torch([spk_id])
            yield Target_id
        else:
            wav_name = tar_wav.split('/')[-1]
            Target_name = wav_name.split('_')[0]
            yield Target_name
            Target_id = label_encoder.encode_sequence_torch([Target_name])
            yield Target_id
        if 'Clean_' in ID:
            Method_id = torch.tensor([0])
        elif 'PPG_' in ID:
            Method_id = torch.tensor([1])
        elif 'Again_' in ID:
            Method_id = torch.tensor([2])
        elif 'VQVC_' in ID:
            Method_id = torch.tensor([3])
        elif 'VQVCp_' in ID:
            Method_id = torch.tensor([4])
        elif 'Diff_' in ID:
            Method_id = torch.tensor([5])
        elif 'DDDM_' in ID:
            Method_id = torch.tensor([6])
        elif 'FreeVC_' in ID:
            Method_id = torch.tensor([7])
        else:
            Method_id = torch.tensor([8])
        yield Method_id

    sb.dataio.dataset.add_dynamic_item(datasets, target_method_label_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="spk_id",
    )

    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "spk_id_encoded", "tar_sig", "Target_id", "Target_name", "Method_id"])# VC_encoded

    return train_data, valid_data


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)

    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams['pretrain_mode'] not in ('ecapa', 'trident'):
        raise ValueError("pretrain_mode must be 'ecapa' or 'trident'")

    speaker_brain = SpeakerBrain(
        modules=hparams["N_modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    if hparams['pretrain_mode'] == 'trident':
        pretrain = Pretrainer(
                    loadables = {'model':speaker_brain.modules.embedding_model,
                                'Target_Projection':speaker_brain.modules.Target_Projection,
                                'Method_Projection':speaker_brain.modules.Method_Projection,
                                'Source_Projection':speaker_brain.modules.Source_Projection,
                                'Target_classifier':speaker_brain.modules.Target_classifier,
                                'Method_classifier':speaker_brain.modules.Method_classifier,
                                'Source_classifier':speaker_brain.modules.Source_classifier,
                                },
                    paths = {'model':hparams['embedding_model_path'],
                            'Target_Projection':hparams['Target_Projection_path'],
                            'Method_Projection':hparams['Method_Projection_path'],
                            'Source_Projection':hparams['Source_Projection_path'],
                            'Target_classifier':hparams['Target_classifier_path'],
                            'Method_classifier':hparams['Method_classifier_path'],
                            'Source_classifier':hparams['Source_classifier_path'],
                            }    
        )
    else:
        pretrain = Pretrainer(
            loadables = {'model':speaker_brain.modules.embedding_model},
            paths = {'model': hparams['embedding_model_path']}
        )
    pretrain.collect_files()
    pretrain.load_collected()

    train_data, valid_data = dataio_prep(hparams)

    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )
