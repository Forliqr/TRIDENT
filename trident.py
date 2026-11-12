import torchaudio
import os
import torch
from speechbrain.pretrained import Pretrained

class Encoder(Pretrained):
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch(self, wavs, tarwavs, wav_lens=None, tarwav_lens=None, normalize=False):
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        if len(tarwavs.shape) == 1:
            tarwavs = tarwavs.unsqueeze(0)

        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        if tarwav_lens is None:
            tarwav_lens = torch.ones(tarwavs.shape[0], device=self.device)

        feats1 = self.mods.compute_features(wavs.to(self.device))
        feats2 = self.mods.compute_features(tarwavs.to(self.device))

        feats1 = self.mods.mean_var_norm(feats1, wav_lens)
        feats2 = self.mods.mean_var_norm(feats2, tarwav_lens)

        embeddings = self.mods.embedding_model.module.inference(feats1, feats2)     

        Method_embeddings, un_embeddings = self.mods.Method_Projection(embeddings)

        embeddings = self.mods.Source_Projection(embeddings, un_embeddings)

        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings

    def encode_batch_ref(self, wavs, tarwavs, wav_lens=None, tarwav_lens=None, normalize=False):
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        if len(tarwavs.shape) == 1:
            tarwavs = tarwavs.unsqueeze(0)

        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)
        if tarwav_lens is None:
            tarwav_lens = torch.ones(tarwavs.shape[0], device=self.device)

        feats1 = self.mods.compute_features(wavs.to(self.device))
        feats2 = self.mods.compute_features(tarwavs.to(self.device))

        feats1 = self.mods.mean_var_norm(feats1, wav_lens)
        feats2 = self.mods.mean_var_norm(feats2, tarwav_lens)

        embeddings = self.mods.embedding_model.module.inference(feats1, feats2)   

        Method_embeddings, un_embeddings = self.mods.Method_Projection(embeddings)

        embeddings = self.mods.Source_Projection(embeddings, un_embeddings)

        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings, torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings

    def encode_file(self, path, tarvp):
        waveform = self.load_audio(path)

        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        emb = self.encode_batch(batch, tarvp, rel_length)
       
        return emb

    def forward(self, wavs, tarwavs, wav_lens=None, tarwav_lens=None):
        return self.encode_batch(wavs=wavs, tarwavs=tarwavs)
