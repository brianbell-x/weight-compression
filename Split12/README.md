# Split12

Byte-split lossless compression for BF16 LLM weights — the current proven
method in this ledger.

## Results

Full scan of `zai-org/GLM-5.2` (59,509 BF16 tensors, round-tripped bit-for-bit):

- **24.967% byte-split format** (12.005 bits/weight) — the shipped, verified format
- **30.168% K15 charged-format estimate** (11.173 bits/weight) — estimate, not yet a codec

## Verify

```bash
# from the repo root
uv sync
uv run Split12/verify.py <org>/<repo>
```

The verifier streams the model shard by shard from Hugging Face and checks
losslessness plus decoded reduction. `ALL_BF16_TENSORS_LOSSLESS: true` is the
whole point. It intentionally omits accounting-only layouts it does not decode.

## References

```bibtex
@article{hershcovitch2024foundation,
  title         = {Lossless and Near-Lossless Compression for Foundation Models},
  author        = {Hershcovitch, Moshik and Choshen, Leshem and Wood, Andrew and
                   Ennmouri, Ilias and Chin, Peter and Sundararaman, Swaminathan
                   and Harnik, Danny},
  year          = {2024},
  eprint        = {2404.15198},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2404.15198}
}

@article{liguori2024lossless,
  title         = {From a Lossless ({\textasciitilde}1.5:1) Compression Algorithm
                   for {Llama2} 7B Weights to Variable Precision, Variable Range,
                   Compressed Numeric Data Types for {CNNs} and {LLMs}},
  author        = {Liguori, Vincenzo},
  year          = {2024},
  eprint        = {2404.10896},
  archiveprefix = {arXiv},
  primaryclass  = {cs.AR},
  url           = {https://arxiv.org/abs/2404.10896}
}

@article{hao2024neuzip,
  title         = {{NeuZip}: Memory-Efficient Training and Inference with Dynamic
                   Compression of Neural Networks},
  author        = {Hao, Yongchang and Cao, Yanshuai and Mou, Lili},
  year          = {2024},
  eprint        = {2410.20650},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2410.20650}
}

@article{hershcovitch2024zipnn,
  title         = {{ZipNN}: Lossless Compression for {AI} Models},
  author        = {Hershcovitch, Moshik and Wood, Andrew and Choshen, Leshem and
                   Girmonsky, Guy and Leibovitz, Roy and Ennmouri, Ilias and
                   Malka, Michal and Chin, Peter and Sundararaman, Swaminathan and
                   Harnik, Danny},
  year          = {2024},
  eprint        = {2411.05239},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2411.05239v2}
}

@article{yubeaton2025huffllm,
  title         = {{Huff-LLM}: End-to-End Lossless Compression for Efficient
                   {LLM} Inference},
  author        = {Yubeaton, Patrick and Mahmoud, Tareq and Naga, Shehab and
                   Taheri, Pooria and Xia, Tianhua and George, Arun and
                   Khalil, Yasmein and Zhang, Sai Qian and Joshi, Siddharth and
                   Hegde, Chinmay and Garg, Siddharth},
  year          = {2025},
  eprint        = {2502.00922},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2502.00922}
}

@article{zhang2025dfloat11,
  title         = {70\% Size, 100\% Accuracy: Lossless {LLM} Compression for
                   Efficient {GPU} Inference via Dynamic-Length Float ({DFloat11})},
  author        = {Zhang, Tianyi and Hariri, Mohsen and Zhong, Shaochen and
                   Chaudhary, Vipin and Sui, Yang and Hu, Xia and
                   Shrivastava, Anshumali},
  year          = {2025},
  eprint        = {2504.11651},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2504.11651v3}
}

@article{yang2025ecf8,
  title         = {To Compress or Not? Pushing the Frontier of Lossless {GenAI}
                   Model Weights Compression with Exponent Concentration},
  author        = {Yang, Zeyu and Zhang, Tianyi and Xie, Jianwen and Li, Chuan and
                   Xu, Zhaozhuo and Shrivastava, Anshumali},
  year          = {2025},
  eprint        = {2510.02676},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.02676}
}

@inproceedings{liberman2026nf12,
  title     = {{NyoomFloat12}: Lossless 12-bit Weight Compression for
               Post-Training Inference},
  author    = {Liberman, Sylvie and Zhang, Tianyi and Fu, Daniel Y.},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=yYFj1E1mSZ}
}

@article{agrawal2026dual,
  title         = {Dual Length Codes for Lossless Compression of {BFloat16}},
  author        = {Agrawal, Aditya and Magyar, Albert and Eswaraiah, Hiteshwar
                   and Sheridan, Patrick and Janedula, Pradeep and
                   Venkatesan, Ravi Krishnan and Nair, Krishna and Iyer, Ravi},
  year          = {2026},
  eprint        = {2602.17849v1},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  note          = {Version 2 retitled and retargeted the method to e4m3},
  url           = {https://arxiv.org/abs/2602.17849v1}
}

@article{fan2026zipserv,
  title         = {{ZipServ}: Fast and Memory-Efficient {LLM} Inference with
                   Hardware-Aware Lossless Compression},
  author        = {Fan, Ruibo and Yu, Xiangrui and Pan, Xinglin and Li, Zeyu and
                   Luo, Weile and Wang, Qiang and Wang, Wei and Chu, Xiaowen},
  year          = {2026},
  eprint        = {2603.17435},
  archiveprefix = {arXiv},
  primaryclass  = {cs.DC},
  url           = {https://arxiv.org/abs/2603.17435v1}
}

@article{sun2026lexi,
  title         = {{LEXI}: Lossless Exponent Coding for Efficient Inter-Chiplet
                   Communication in Hybrid {LLMs}},
  author        = {Sun, Miao and Kanani, Alish and Shroff, Kaushik and
                   Ogras, Umit},
  year          = {2026},
  eprint        = {2603.15589},
  archiveprefix = {arXiv},
  primaryclass  = {cs.AR},
  url           = {https://arxiv.org/abs/2603.15589}
}

@techreport{nikulin2026unweight,
  title       = {{Unweight}: Lossless {MLP} Weight Compression for {LLM}
                 Inference},
  author      = {Nikulin, Ivan},
  institution = {Cloudflare Research},
  number      = {Cf-TR-2026.04.v1},
  year        = {2026},
  month       = apr,
  url         = {https://research.cloudflare.com/papers/unweight-2026.pdf}
}

@article{yang2026enec,
  title         = {{ENEC}: A Lossless {AI} Model Compression Method Enabling
                   Fast Inference on Ascend {NPUs}},
  author        = {Yang, Jinwu and Wu, Jiaan and Liu, Zedong and Ma, Xinyang and
                   Zhao, Hairui and Gu, Yida and Huang, Yuanhong and Liu, Xingchen
                   and Huang, Wenjing and Wei, Zheng and Xing, Jing and Ma, Yili
                   and Zhang, Qingyi and An, Baoyi and Hu, Zhongzhe and Liu,
                   Shaoteng and Zhu, Xia and Lu, Jiaxun and Tan, Guangming and
                   Tao, Dingwen},
  year          = {2026},
  eprint        = {2604.03298},
  archiveprefix = {arXiv},
  primaryclass  = {cs.DC},
  url           = {https://arxiv.org/abs/2604.03298}
}

@article{guo2026splitzip,
  title         = {{SplitZip}: Ultra Fast Lossless {KV} Compression for
                   Disaggregated {LLM} Serving},
  author        = {Guo, Yipin and Joshi, Siddharth},
  year          = {2026},
  eprint        = {2605.01708},
  archiveprefix = {arXiv},
  primaryclass  = {cs.DC},
  url           = {https://arxiv.org/abs/2605.01708}
}

@article{kamath2026ibp,
  title         = {Reducing the {GPU} Memory Bottleneck with Lossless
                   Compression for {ML} -- Extended},
  author        = {Kamath, Aditya K. and Krishnamurthy, Arvind and Canini, Marco
                   and Peter, Simon},
  year          = {2026},
  eprint        = {2605.30728},
  archiveprefix = {arXiv},
  primaryclass  = {cs.DC},
  url           = {https://arxiv.org/abs/2605.30728}
}

@article{tan2026shannon,
  title         = {Approaching Shannon Bound with Lossless {LLM} Weight
                   Compression},
  author        = {Tan, Hongshi and Chen, Yao and Alonso, Gustavo and
                   Wong, Weng-Fai and He, Bingsheng},
  year          = {2026},
  eprint        = {2606.15789},
  archiveprefix = {arXiv},
  primaryclass  = {cs.DC},
  url           = {https://arxiv.org/abs/2606.15789}
}
```
