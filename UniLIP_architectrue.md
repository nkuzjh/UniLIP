# model
# UniLIP_InternVLForCausalLM(
  ## (model): UniLIP_InternVLModel(
   ### (vision_tower): InternVisionModel(
      (embeddings): InternVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
      )
      (encoder): InternVisionEncoder(
        (layers): ModuleList(
          (0): InternVisionEncoderLayer(
            (attn): InternAttention(
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (inner_attn): FlashAttention()
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (mlp): InternMLP(
              (act): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (drop_path1): Identity()
            (drop_path2): Identity()
          )
          (1-23): 23 x InternVisionEncoderLayer(
            (attn): InternAttention(
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (inner_attn): FlashAttention()
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (mlp): InternMLP(
              (act): GELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (drop_path1): DropPath(drop_prob=0.000)
            (drop_path2): DropPath(drop_prob=0.000)
          )
        )
      )
    )
  ### (multi_modal_projector): Sequential(
      (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
      (1): Linear(in_features=4096, out_features=896, bias=True)
      (2): GELU(approximate='none')
      (3): Linear(in_features=896, out_features=896, bias=True)
    )
  ### (language_model): Qwen2Model(
      (embed_tokens): Embedding(151678, 896)
      (layers): ModuleList(
        (0-23): 24 x Qwen2DecoderLayer(
          (self_attn): Qwen2Attention(
            (q_proj): Linear(in_features=896, out_features=896, bias=True)
            (k_proj): Linear(in_features=896, out_features=128, bias=True)
            (v_proj): Linear(in_features=896, out_features=128, bias=True)
            (o_proj): Linear(in_features=896, out_features=896, bias=False)
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
            (up_proj): Linear(in_features=896, out_features=4864, bias=False)
            (down_proj): Linear(in_features=4864, out_features=896, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((896,), eps=1e-06)
      (rotary_emb): Qwen2RotaryEmbedding()
    )
  ### (dit): SanaTransformer2DModel(
      (patch_embed): PatchEmbed(
        (proj): Conv2d(32, 1152, kernel_size=(1, 1), stride=(1, 1))
      )
      (time_embed): AdaLayerNormSingle(
        (emb): PixArtAlphaCombinedTimestepSizeEmbeddings(
          (time_proj): Timesteps()
          (timestep_embedder): TimestepEmbedding(
            (linear_1): Linear(in_features=256, out_features=1152, bias=True)
            (act): SiLU()
            (linear_2): Linear(in_features=1152, out_features=1152, bias=True)
          )
        )
        (silu): SiLU()
        (linear): Linear(in_features=1152, out_features=6912, bias=True)
      )
      (caption_projection): PixArtAlphaTextProjection(
        (linear_1): Linear(in_features=2304, out_features=1152, bias=True)
        (act_1): GELU(approximate='tanh')
        (linear_2): Linear(in_features=1152, out_features=1152, bias=True)
      )
      (caption_norm): RMSNorm()
      (transformer_blocks): ModuleList(
        (0-27): 28 x SanaTransformerBlock(
          (norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=False)
          (attn1): Attention(
            (to_q): Linear(in_features=1152, out_features=1152, bias=False)
            (to_k): Linear(in_features=1152, out_features=1152, bias=False)
            (to_v): Linear(in_features=1152, out_features=1152, bias=False)
            (to_out): ModuleList(
              (0): Linear(in_features=1152, out_features=1152, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=False)
          (attn2): Attention(
            (to_q): Linear(in_features=1152, out_features=1152, bias=True)
            (to_k): Linear(in_features=1152, out_features=1152, bias=True)
            (to_v): Linear(in_features=1152, out_features=1152, bias=True)
            (to_out): ModuleList(
              (0): Linear(in_features=1152, out_features=1152, bias=True)
              (1): Dropout(p=0.0, inplace=False)
            )
          )
          (ff): GLUMBConv(
            (nonlinearity): SiLU()
            (conv_inverted): Conv2d(1152, 5760, kernel_size=(1, 1), stride=(1, 1))
            (conv_depth): Conv2d(5760, 5760, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=5760)
            (conv_point): Conv2d(2880, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
        )
      )
      (norm_out): SanaModulatedNorm(
        (norm): LayerNorm((1152,), eps=1e-06, elementwise_affine=False)
      )
      (proj_out): Linear(in_features=1152, out_features=32, bias=True)
    )
  ### (vae_decoder): DCAE_Decoder(
      (decoder): Decoder(
        (conv_in): Conv2d(32, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (up_blocks): ModuleList(
          (0): Sequential(
            (0): DCUpBlock2d(
              (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
            (2): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
            (3): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
          )
          (1): Sequential(
            (0): DCUpBlock2d(
              (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
            (2): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
            (3): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
          )
          (2): Sequential(
            (0): DCUpBlock2d(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
            (2): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
            (3): ResBlock(
              (nonlinearity): SiLU()
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
              (norm): RMSNorm()
            )
          )
          (3): Sequential(
            (0): DCUpBlock2d(
              (conv): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=512, out_features=512, bias=False)
                (to_k): Linear(in_features=512, out_features=512, bias=False)
                (to_v): Linear(in_features=512, out_features=512, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(1536, 1536, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1536, bias=False)
                    (proj_out): Conv2d(1536, 1536, kernel_size=(1, 1), stride=(1, 1), groups=48, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=1024, out_features=512, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(512, 4096, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(4096, 4096, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4096)
                (conv_point): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
            (2): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=512, out_features=512, bias=False)
                (to_k): Linear(in_features=512, out_features=512, bias=False)
                (to_v): Linear(in_features=512, out_features=512, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(1536, 1536, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1536, bias=False)
                    (proj_out): Conv2d(1536, 1536, kernel_size=(1, 1), stride=(1, 1), groups=48, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=1024, out_features=512, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(512, 4096, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(4096, 4096, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4096)
                (conv_point): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
            (3): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=512, out_features=512, bias=False)
                (to_k): Linear(in_features=512, out_features=512, bias=False)
                (to_v): Linear(in_features=512, out_features=512, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(1536, 1536, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1536, bias=False)
                    (proj_out): Conv2d(1536, 1536, kernel_size=(1, 1), stride=(1, 1), groups=48, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=1024, out_features=512, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(512, 4096, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(4096, 4096, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=4096)
                (conv_point): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
          )
          (4): Sequential(
            (0): DCUpBlock2d(
              (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=1024, out_features=1024, bias=False)
                (to_k): Linear(in_features=1024, out_features=1024, bias=False)
                (to_v): Linear(in_features=1024, out_features=1024, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(3072, 3072, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=3072, bias=False)
                    (proj_out): Conv2d(3072, 3072, kernel_size=(1, 1), stride=(1, 1), groups=96, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=2048, out_features=1024, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(1024, 8192, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(8192, 8192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8192)
                (conv_point): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
            (2): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=1024, out_features=1024, bias=False)
                (to_k): Linear(in_features=1024, out_features=1024, bias=False)
                (to_v): Linear(in_features=1024, out_features=1024, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(3072, 3072, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=3072, bias=False)
                    (proj_out): Conv2d(3072, 3072, kernel_size=(1, 1), stride=(1, 1), groups=96, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=2048, out_features=1024, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(1024, 8192, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(8192, 8192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8192)
                (conv_point): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
            (3): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=1024, out_features=1024, bias=False)
                (to_k): Linear(in_features=1024, out_features=1024, bias=False)
                (to_v): Linear(in_features=1024, out_features=1024, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(3072, 3072, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=3072, bias=False)
                    (proj_out): Conv2d(3072, 3072, kernel_size=(1, 1), stride=(1, 1), groups=96, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=2048, out_features=1024, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(1024, 8192, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(8192, 8192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8192)
                (conv_point): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
          )
          (5): Sequential(
            (0): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=1024, out_features=1024, bias=False)
                (to_k): Linear(in_features=1024, out_features=1024, bias=False)
                (to_v): Linear(in_features=1024, out_features=1024, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(3072, 3072, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=3072, bias=False)
                    (proj_out): Conv2d(3072, 3072, kernel_size=(1, 1), stride=(1, 1), groups=96, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=2048, out_features=1024, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(1024, 8192, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(8192, 8192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8192)
                (conv_point): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
            (1): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=1024, out_features=1024, bias=False)
                (to_k): Linear(in_features=1024, out_features=1024, bias=False)
                (to_v): Linear(in_features=1024, out_features=1024, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(3072, 3072, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=3072, bias=False)
                    (proj_out): Conv2d(3072, 3072, kernel_size=(1, 1), stride=(1, 1), groups=96, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=2048, out_features=1024, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(1024, 8192, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(8192, 8192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8192)
                (conv_point): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
            (2): EfficientViTBlock(
              (attn): SanaMultiscaleLinearAttention(
                (to_q): Linear(in_features=1024, out_features=1024, bias=False)
                (to_k): Linear(in_features=1024, out_features=1024, bias=False)
                (to_v): Linear(in_features=1024, out_features=1024, bias=False)
                (to_qkv_multiscale): ModuleList(
                  (0): SanaMultiscaleAttentionProjection(
                    (proj_in): Conv2d(3072, 3072, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=3072, bias=False)
                    (proj_out): Conv2d(3072, 3072, kernel_size=(1, 1), stride=(1, 1), groups=96, bias=False)
                  )
                )
                (nonlinearity): ReLU()
                (to_out): Linear(in_features=2048, out_features=1024, bias=False)
                (norm_out): RMSNorm()
              )
              (conv_out): GLUMBConv(
                (nonlinearity): SiLU()
                (conv_inverted): Conv2d(1024, 8192, kernel_size=(1, 1), stride=(1, 1))
                (conv_depth): Conv2d(8192, 8192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8192)
                (conv_point): Conv2d(4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (norm): RMSNorm()
              )
            )
          )
        )
        (norm_out): RMSNorm()
        (conv_act): ReLU()
        (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (down_blocks): ModuleList(
        (0-2): 3 x ResBlock(
          (mlp): Sequential(
            (0): LayerNorm((896,), eps=1e-06, elementwise_affine=True)
            (1): Linear(in_features=896, out_features=896, bias=True)
            (2): GELU(approximate='none')
            (3): Linear(in_features=896, out_features=896, bias=True)
          )
        )
      )
      (down_mlp): Sequential(
        (0): LayerNorm((896,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=896, out_features=32, bias=True)
        (2): GELU(approximate='none')
        (3): Linear(in_features=32, out_features=32, bias=True)
      )
    )
  ### (llm_connector): Qwen2Model(
      (layers): ModuleList(
        (0-5): 6 x Qwen2DecoderLayer(
          (self_attn): Qwen2Attention(
            (q_proj): Linear(in_features=896, out_features=896, bias=True)
            (k_proj): Linear(in_features=896, out_features=128, bias=True)
            (v_proj): Linear(in_features=896, out_features=128, bias=True)
            (o_proj): Linear(in_features=896, out_features=896, bias=False)
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=896, out_features=4864, bias=False)
            (up_proj): Linear(in_features=896, out_features=4864, bias=False)
            (down_proj): Linear(in_features=4864, out_features=896, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((896,), eps=1e-06)
      (rotary_emb): Qwen2RotaryEmbedding()
    )
  ### (projector): Linear(in_features=896, out_features=2304, bias=True)
  )
  ## (lm_head): Linear(in_features=896, out_features=151678, bias=False)
)





# Total parameters: 1649898211
# Trainable parameters: 683519136
##      trainable params:  model.latent_queries
##      trainable params:  model.dit.scale_shift_table
     trainable params:  model.dit.patch_embed.proj.weight
     trainable params:  model.dit.patch_embed.proj.bias
     trainable params:  model.dit.time_embed.emb.timestep_embedder.linear_1.weight
     trainable params:  model.dit.time_embed.emb.timestep_embedder.linear_1.bias
     trainable params:  model.dit.time_embed.emb.timestep_embedder.linear_2.weight
     trainable params:  model.dit.time_embed.emb.timestep_embedder.linear_2.bias
     trainable params:  model.dit.time_embed.linear.weight
     trainable params:  model.dit.time_embed.linear.bias
     trainable params:  model.dit.caption_projection.linear_1.weight
     trainable params:  model.dit.caption_projection.linear_1.bias
     trainable params:  model.dit.caption_projection.linear_2.weight
     trainable params:  model.dit.caption_projection.linear_2.bias
     trainable params:  model.dit.caption_norm.weight
     trainable params:  model.dit.transformer_blocks.0.scale_shift_table
     trainable params:  model.dit.transformer_blocks.0.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.0.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.0.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.0.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.0.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.0.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.0.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.0.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.0.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.0.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.0.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.0.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.0.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.0.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.0.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.0.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.0.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.0.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.1.scale_shift_table
     trainable params:  model.dit.transformer_blocks.1.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.1.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.1.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.1.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.1.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.1.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.1.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.1.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.1.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.1.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.1.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.1.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.1.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.1.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.1.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.1.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.1.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.1.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.2.scale_shift_table
     trainable params:  model.dit.transformer_blocks.2.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.2.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.2.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.2.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.2.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.2.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.2.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.2.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.2.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.2.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.2.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.2.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.2.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.2.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.2.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.2.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.2.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.2.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.3.scale_shift_table
     trainable params:  model.dit.transformer_blocks.3.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.3.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.3.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.3.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.3.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.3.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.3.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.3.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.3.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.3.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.3.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.3.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.3.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.3.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.3.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.3.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.3.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.3.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.4.scale_shift_table
     trainable params:  model.dit.transformer_blocks.4.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.4.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.4.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.4.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.4.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.4.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.4.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.4.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.4.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.4.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.4.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.4.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.4.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.4.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.4.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.4.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.4.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.4.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.5.scale_shift_table
     trainable params:  model.dit.transformer_blocks.5.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.5.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.5.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.5.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.5.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.5.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.5.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.5.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.5.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.5.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.5.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.5.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.5.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.5.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.5.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.5.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.5.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.5.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.6.scale_shift_table
     trainable params:  model.dit.transformer_blocks.6.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.6.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.6.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.6.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.6.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.6.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.6.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.6.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.6.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.6.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.6.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.6.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.6.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.6.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.6.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.6.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.6.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.6.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.7.scale_shift_table
     trainable params:  model.dit.transformer_blocks.7.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.7.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.7.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.7.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.7.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.7.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.7.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.7.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.7.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.7.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.7.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.7.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.7.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.7.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.7.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.7.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.7.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.7.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.8.scale_shift_table
     trainable params:  model.dit.transformer_blocks.8.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.8.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.8.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.8.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.8.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.8.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.8.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.8.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.8.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.8.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.8.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.8.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.8.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.8.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.8.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.8.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.8.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.8.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.9.scale_shift_table
     trainable params:  model.dit.transformer_blocks.9.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.9.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.9.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.9.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.9.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.9.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.9.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.9.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.9.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.9.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.9.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.9.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.9.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.9.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.9.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.9.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.9.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.9.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.10.scale_shift_table
     trainable params:  model.dit.transformer_blocks.10.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.10.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.10.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.10.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.10.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.10.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.10.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.10.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.10.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.10.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.10.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.10.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.10.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.10.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.10.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.10.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.10.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.10.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.11.scale_shift_table
     trainable params:  model.dit.transformer_blocks.11.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.11.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.11.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.11.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.11.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.11.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.11.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.11.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.11.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.11.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.11.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.11.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.11.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.11.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.11.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.11.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.11.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.11.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.12.scale_shift_table
     trainable params:  model.dit.transformer_blocks.12.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.12.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.12.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.12.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.12.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.12.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.12.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.12.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.12.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.12.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.12.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.12.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.12.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.12.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.12.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.12.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.12.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.12.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.13.scale_shift_table
     trainable params:  model.dit.transformer_blocks.13.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.13.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.13.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.13.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.13.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.13.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.13.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.13.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.13.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.13.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.13.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.13.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.13.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.13.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.13.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.13.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.13.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.13.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.14.scale_shift_table
     trainable params:  model.dit.transformer_blocks.14.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.14.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.14.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.14.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.14.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.14.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.14.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.14.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.14.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.14.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.14.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.14.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.14.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.14.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.14.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.14.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.14.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.14.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.15.scale_shift_table
     trainable params:  model.dit.transformer_blocks.15.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.15.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.15.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.15.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.15.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.15.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.15.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.15.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.15.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.15.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.15.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.15.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.15.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.15.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.15.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.15.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.15.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.15.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.16.scale_shift_table
     trainable params:  model.dit.transformer_blocks.16.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.16.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.16.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.16.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.16.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.16.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.16.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.16.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.16.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.16.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.16.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.16.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.16.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.16.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.16.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.16.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.16.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.16.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.17.scale_shift_table
     trainable params:  model.dit.transformer_blocks.17.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.17.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.17.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.17.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.17.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.17.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.17.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.17.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.17.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.17.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.17.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.17.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.17.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.17.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.17.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.17.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.17.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.17.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.18.scale_shift_table
     trainable params:  model.dit.transformer_blocks.18.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.18.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.18.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.18.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.18.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.18.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.18.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.18.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.18.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.18.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.18.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.18.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.18.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.18.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.18.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.18.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.18.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.18.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.19.scale_shift_table
     trainable params:  model.dit.transformer_blocks.19.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.19.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.19.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.19.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.19.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.19.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.19.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.19.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.19.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.19.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.19.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.19.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.19.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.19.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.19.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.19.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.19.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.19.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.20.scale_shift_table
     trainable params:  model.dit.transformer_blocks.20.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.20.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.20.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.20.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.20.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.20.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.20.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.20.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.20.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.20.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.20.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.20.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.20.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.20.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.20.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.20.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.20.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.20.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.21.scale_shift_table
     trainable params:  model.dit.transformer_blocks.21.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.21.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.21.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.21.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.21.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.21.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.21.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.21.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.21.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.21.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.21.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.21.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.21.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.21.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.21.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.21.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.21.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.21.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.22.scale_shift_table
     trainable params:  model.dit.transformer_blocks.22.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.22.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.22.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.22.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.22.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.22.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.22.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.22.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.22.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.22.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.22.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.22.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.22.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.22.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.22.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.22.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.22.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.22.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.23.scale_shift_table
     trainable params:  model.dit.transformer_blocks.23.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.23.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.23.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.23.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.23.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.23.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.23.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.23.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.23.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.23.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.23.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.23.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.23.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.23.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.23.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.23.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.23.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.23.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.24.scale_shift_table
     trainable params:  model.dit.transformer_blocks.24.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.24.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.24.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.24.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.24.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.24.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.24.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.24.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.24.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.24.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.24.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.24.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.24.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.24.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.24.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.24.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.24.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.24.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.25.scale_shift_table
     trainable params:  model.dit.transformer_blocks.25.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.25.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.25.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.25.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.25.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.25.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.25.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.25.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.25.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.25.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.25.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.25.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.25.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.25.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.25.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.25.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.25.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.25.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.26.scale_shift_table
     trainable params:  model.dit.transformer_blocks.26.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.26.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.26.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.26.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.26.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.26.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.26.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.26.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.26.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.26.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.26.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.26.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.26.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.26.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.26.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.26.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.26.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.26.ff.conv_point.weight
     trainable params:  model.dit.transformer_blocks.27.scale_shift_table
     trainable params:  model.dit.transformer_blocks.27.attn1.to_q.weight
     trainable params:  model.dit.transformer_blocks.27.attn1.to_k.weight
     trainable params:  model.dit.transformer_blocks.27.attn1.to_v.weight
     trainable params:  model.dit.transformer_blocks.27.attn1.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.27.attn1.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.27.attn2.to_q.weight
     trainable params:  model.dit.transformer_blocks.27.attn2.to_q.bias
     trainable params:  model.dit.transformer_blocks.27.attn2.to_k.weight
     trainable params:  model.dit.transformer_blocks.27.attn2.to_k.bias
     trainable params:  model.dit.transformer_blocks.27.attn2.to_v.weight
     trainable params:  model.dit.transformer_blocks.27.attn2.to_v.bias
     trainable params:  model.dit.transformer_blocks.27.attn2.to_out.0.weight
     trainable params:  model.dit.transformer_blocks.27.attn2.to_out.0.bias
     trainable params:  model.dit.transformer_blocks.27.ff.conv_inverted.weight
     trainable params:  model.dit.transformer_blocks.27.ff.conv_inverted.bias
     trainable params:  model.dit.transformer_blocks.27.ff.conv_depth.weight
     trainable params:  model.dit.transformer_blocks.27.ff.conv_depth.bias
     trainable params:  model.dit.transformer_blocks.27.ff.conv_point.weight
     trainable params:  model.dit.proj_out.weight
     trainable params:  model.dit.proj_out.bias
##      trainable params:  model.llm_connector.layers.0.self_attn.q_proj.weight
     trainable params:  model.llm_connector.layers.0.self_attn.q_proj.bias
     trainable params:  model.llm_connector.layers.0.self_attn.k_proj.weight
     trainable params:  model.llm_connector.layers.0.self_attn.k_proj.bias
     trainable params:  model.llm_connector.layers.0.self_attn.v_proj.weight
     trainable params:  model.llm_connector.layers.0.self_attn.v_proj.bias
     trainable params:  model.llm_connector.layers.0.self_attn.o_proj.weight
     trainable params:  model.llm_connector.layers.0.mlp.gate_proj.weight
     trainable params:  model.llm_connector.layers.0.mlp.up_proj.weight
     trainable params:  model.llm_connector.layers.0.mlp.down_proj.weight
     trainable params:  model.llm_connector.layers.0.input_layernorm.weight
     trainable params:  model.llm_connector.layers.0.post_attention_layernorm.weight
     trainable params:  model.llm_connector.layers.1.self_attn.q_proj.weight
     trainable params:  model.llm_connector.layers.1.self_attn.q_proj.bias
     trainable params:  model.llm_connector.layers.1.self_attn.k_proj.weight
     trainable params:  model.llm_connector.layers.1.self_attn.k_proj.bias
     trainable params:  model.llm_connector.layers.1.self_attn.v_proj.weight
     trainable params:  model.llm_connector.layers.1.self_attn.v_proj.bias
     trainable params:  model.llm_connector.layers.1.self_attn.o_proj.weight
     trainable params:  model.llm_connector.layers.1.mlp.gate_proj.weight
     trainable params:  model.llm_connector.layers.1.mlp.up_proj.weight
     trainable params:  model.llm_connector.layers.1.mlp.down_proj.weight
     trainable params:  model.llm_connector.layers.1.input_layernorm.weight
     trainable params:  model.llm_connector.layers.1.post_attention_layernorm.weight
     trainable params:  model.llm_connector.layers.2.self_attn.q_proj.weight
     trainable params:  model.llm_connector.layers.2.self_attn.q_proj.bias
     trainable params:  model.llm_connector.layers.2.self_attn.k_proj.weight
     trainable params:  model.llm_connector.layers.2.self_attn.k_proj.bias
     trainable params:  model.llm_connector.layers.2.self_attn.v_proj.weight
     trainable params:  model.llm_connector.layers.2.self_attn.v_proj.bias
     trainable params:  model.llm_connector.layers.2.self_attn.o_proj.weight
     trainable params:  model.llm_connector.layers.2.mlp.gate_proj.weight
     trainable params:  model.llm_connector.layers.2.mlp.up_proj.weight
     trainable params:  model.llm_connector.layers.2.mlp.down_proj.weight
     trainable params:  model.llm_connector.layers.2.input_layernorm.weight
     trainable params:  model.llm_connector.layers.2.post_attention_layernorm.weight
     trainable params:  model.llm_connector.layers.3.self_attn.q_proj.weight
     trainable params:  model.llm_connector.layers.3.self_attn.q_proj.bias
     trainable params:  model.llm_connector.layers.3.self_attn.k_proj.weight
     trainable params:  model.llm_connector.layers.3.self_attn.k_proj.bias
     trainable params:  model.llm_connector.layers.3.self_attn.v_proj.weight
     trainable params:  model.llm_connector.layers.3.self_attn.v_proj.bias
     trainable params:  model.llm_connector.layers.3.self_attn.o_proj.weight
     trainable params:  model.llm_connector.layers.3.mlp.gate_proj.weight
     trainable params:  model.llm_connector.layers.3.mlp.up_proj.weight
     trainable params:  model.llm_connector.layers.3.mlp.down_proj.weight
     trainable params:  model.llm_connector.layers.3.input_layernorm.weight
     trainable params:  model.llm_connector.layers.3.post_attention_layernorm.weight
     trainable params:  model.llm_connector.layers.4.self_attn.q_proj.weight
     trainable params:  model.llm_connector.layers.4.self_attn.q_proj.bias
     trainable params:  model.llm_connector.layers.4.self_attn.k_proj.weight
     trainable params:  model.llm_connector.layers.4.self_attn.k_proj.bias
     trainable params:  model.llm_connector.layers.4.self_attn.v_proj.weight
     trainable params:  model.llm_connector.layers.4.self_attn.v_proj.bias
     trainable params:  model.llm_connector.layers.4.self_attn.o_proj.weight
     trainable params:  model.llm_connector.layers.4.mlp.gate_proj.weight
     trainable params:  model.llm_connector.layers.4.mlp.up_proj.weight
     trainable params:  model.llm_connector.layers.4.mlp.down_proj.weight
     trainable params:  model.llm_connector.layers.4.input_layernorm.weight
     trainable params:  model.llm_connector.layers.4.post_attention_layernorm.weight
     trainable params:  model.llm_connector.layers.5.self_attn.q_proj.weight
     trainable params:  model.llm_connector.layers.5.self_attn.q_proj.bias
     trainable params:  model.llm_connector.layers.5.self_attn.k_proj.weight
     trainable params:  model.llm_connector.layers.5.self_attn.k_proj.bias
     trainable params:  model.llm_connector.layers.5.self_attn.v_proj.weight
     trainable params:  model.llm_connector.layers.5.self_attn.v_proj.bias
     trainable params:  model.llm_connector.layers.5.self_attn.o_proj.weight
     trainable params:  model.llm_connector.layers.5.mlp.gate_proj.weight
     trainable params:  model.llm_connector.layers.5.mlp.up_proj.weight
     trainable params:  model.llm_connector.layers.5.mlp.down_proj.weight
     trainable params:  model.llm_connector.layers.5.input_layernorm.weight
     trainable params:  model.llm_connector.layers.5.post_attention_layernorm.weight
     trainable params:  model.llm_connector.norm.weight
##      trainable params:  model.projector.weight
     trainable params:  model.projector.bias