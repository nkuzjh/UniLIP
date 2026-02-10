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
## Trainable parameters: 683519136
## model.latent_queries
     trainable params:  model.latent_queries
## model.dit
     trainable params:  model.dit.scale_shift_table
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
## model.llm_connector
     trainable params:  model.llm_connector.layers.0.self_attn.q_proj.weight
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
## model.projector
     trainable params:  model.projector.weight
     trainable params:  model.projector.bias



# state_dict.keys()
## 'model.latent_queries'
odict_keys(['model.latent_queries',
## 'model.vision_tower
'model.vision_tower.embeddings.class_embedding', 'model.vision_tower.embeddings.position_embedding', 'model.vision_tower.embeddings.patch_embedding.weight', 'model.vision_tower.embeddings.patch_embedding.bias', 'model.vision_tower.encoder.layers.0.ls1', 'model.vision_tower.encoder.layers.0.ls2', 'model.vision_tower.encoder.layers.0.attn.qkv.weight', 'model.vision_tower.encoder.layers.0.attn.qkv.bias', 'model.vision_tower.encoder.layers.0.attn.proj.weight', 'model.vision_tower.encoder.layers.0.attn.proj.bias', 'model.vision_tower.encoder.layers.0.mlp.fc1.weight', 'model.vision_tower.encoder.layers.0.mlp.fc1.bias', 'model.vision_tower.encoder.layers.0.mlp.fc2.weight', 'model.vision_tower.encoder.layers.0.mlp.fc2.bias', 'model.vision_tower.encoder.layers.0.norm1.weight', 'model.vision_tower.encoder.layers.0.norm1.bias', 'model.vision_tower.encoder.layers.0.norm2.weight', 'model.vision_tower.encoder.layers.0.norm2.bias', 'model.vision_tower.encoder.layers.1.ls1', 'model.vision_tower.encoder.layers.1.ls2', 'model.vision_tower.encoder.layers.1.attn.qkv.weight', 'model.vision_tower.encoder.layers.1.attn.qkv.bias', 'model.vision_tower.encoder.layers.1.attn.proj.weight', 'model.vision_tower.encoder.layers.1.attn.proj.bias', 'model.vision_tower.encoder.layers.1.mlp.fc1.weight', 'model.vision_tower.encoder.layers.1.mlp.fc1.bias', 'model.vision_tower.encoder.layers.1.mlp.fc2.weight', 'model.vision_tower.encoder.layers.1.mlp.fc2.bias', 'model.vision_tower.encoder.layers.1.norm1.weight', 'model.vision_tower.encoder.layers.1.norm1.bias', 'model.vision_tower.encoder.layers.1.norm2.weight', 'model.vision_tower.encoder.layers.1.norm2.bias', 'model.vision_tower.encoder.layers.2.ls1', 'model.vision_tower.encoder.layers.2.ls2', 'model.vision_tower.encoder.layers.2.attn.qkv.weight', 'model.vision_tower.encoder.layers.2.attn.qkv.bias', 'model.vision_tower.encoder.layers.2.attn.proj.weight', 'model.vision_tower.encoder.layers.2.attn.proj.bias', 'model.vision_tower.encoder.layers.2.mlp.fc1.weight', 'model.vision_tower.encoder.layers.2.mlp.fc1.bias', 'model.vision_tower.encoder.layers.2.mlp.fc2.weight', 'model.vision_tower.encoder.layers.2.mlp.fc2.bias', 'model.vision_tower.encoder.layers.2.norm1.weight', 'model.vision_tower.encoder.layers.2.norm1.bias', 'model.vision_tower.encoder.layers.2.norm2.weight', 'model.vision_tower.encoder.layers.2.norm2.bias', 'model.vision_tower.encoder.layers.3.ls1', 'model.vision_tower.encoder.layers.3.ls2', 'model.vision_tower.encoder.layers.3.attn.qkv.weight', 'model.vision_tower.encoder.layers.3.attn.qkv.bias', 'model.vision_tower.encoder.layers.3.attn.proj.weight', 'model.vision_tower.encoder.layers.3.attn.proj.bias', 'model.vision_tower.encoder.layers.3.mlp.fc1.weight', 'model.vision_tower.encoder.layers.3.mlp.fc1.bias', 'model.vision_tower.encoder.layers.3.mlp.fc2.weight', 'model.vision_tower.encoder.layers.3.mlp.fc2.bias', 'model.vision_tower.encoder.layers.3.norm1.weight', 'model.vision_tower.encoder.layers.3.norm1.bias', 'model.vision_tower.encoder.layers.3.norm2.weight', 'model.vision_tower.encoder.layers.3.norm2.bias', 'model.vision_tower.encoder.layers.4.ls1', 'model.vision_tower.encoder.layers.4.ls2', 'model.vision_tower.encoder.layers.4.attn.qkv.weight', 'model.vision_tower.encoder.layers.4.attn.qkv.bias', 'model.vision_tower.encoder.layers.4.attn.proj.weight', 'model.vision_tower.encoder.layers.4.attn.proj.bias', 'model.vision_tower.encoder.layers.4.mlp.fc1.weight', 'model.vision_tower.encoder.layers.4.mlp.fc1.bias', 'model.vision_tower.encoder.layers.4.mlp.fc2.weight', 'model.vision_tower.encoder.layers.4.mlp.fc2.bias', 'model.vision_tower.encoder.layers.4.norm1.weight', 'model.vision_tower.encoder.layers.4.norm1.bias', 'model.vision_tower.encoder.layers.4.norm2.weight', 'model.vision_tower.encoder.layers.4.norm2.bias', 'model.vision_tower.encoder.layers.5.ls1', 'model.vision_tower.encoder.layers.5.ls2', 'model.vision_tower.encoder.layers.5.attn.qkv.weight', 'model.vision_tower.encoder.layers.5.attn.qkv.bias', 'model.vision_tower.encoder.layers.5.attn.proj.weight', 'model.vision_tower.encoder.layers.5.attn.proj.bias', 'model.vision_tower.encoder.layers.5.mlp.fc1.weight', 'model.vision_tower.encoder.layers.5.mlp.fc1.bias', 'model.vision_tower.encoder.layers.5.mlp.fc2.weight', 'model.vision_tower.encoder.layers.5.mlp.fc2.bias', 'model.vision_tower.encoder.layers.5.norm1.weight', 'model.vision_tower.encoder.layers.5.norm1.bias', 'model.vision_tower.encoder.layers.5.norm2.weight', 'model.vision_tower.encoder.layers.5.norm2.bias', 'model.vision_tower.encoder.layers.6.ls1', 'model.vision_tower.encoder.layers.6.ls2', 'model.vision_tower.encoder.layers.6.attn.qkv.weight', 'model.vision_tower.encoder.layers.6.attn.qkv.bias', 'model.vision_tower.encoder.layers.6.attn.proj.weight', 'model.vision_tower.encoder.layers.6.attn.proj.bias', 'model.vision_tower.encoder.layers.6.mlp.fc1.weight', 'model.vision_tower.encoder.layers.6.mlp.fc1.bias', 'model.vision_tower.encoder.layers.6.mlp.fc2.weight', 'model.vision_tower.encoder.layers.6.mlp.fc2.bias', 'model.vision_tower.encoder.layers.6.norm1.weight', 'model.vision_tower.encoder.layers.6.norm1.bias', 'model.vision_tower.encoder.layers.6.norm2.weight', 'model.vision_tower.encoder.layers.6.norm2.bias', 'model.vision_tower.encoder.layers.7.ls1', 'model.vision_tower.encoder.layers.7.ls2', 'model.vision_tower.encoder.layers.7.attn.qkv.weight', 'model.vision_tower.encoder.layers.7.attn.qkv.bias', 'model.vision_tower.encoder.layers.7.attn.proj.weight', 'model.vision_tower.encoder.layers.7.attn.proj.bias', 'model.vision_tower.encoder.layers.7.mlp.fc1.weight', 'model.vision_tower.encoder.layers.7.mlp.fc1.bias', 'model.vision_tower.encoder.layers.7.mlp.fc2.weight', 'model.vision_tower.encoder.layers.7.mlp.fc2.bias', 'model.vision_tower.encoder.layers.7.norm1.weight', 'model.vision_tower.encoder.layers.7.norm1.bias', 'model.vision_tower.encoder.layers.7.norm2.weight', 'model.vision_tower.encoder.layers.7.norm2.bias', 'model.vision_tower.encoder.layers.8.ls1', 'model.vision_tower.encoder.layers.8.ls2', 'model.vision_tower.encoder.layers.8.attn.qkv.weight', 'model.vision_tower.encoder.layers.8.attn.qkv.bias', 'model.vision_tower.encoder.layers.8.attn.proj.weight', 'model.vision_tower.encoder.layers.8.attn.proj.bias', 'model.vision_tower.encoder.layers.8.mlp.fc1.weight', 'model.vision_tower.encoder.layers.8.mlp.fc1.bias', 'model.vision_tower.encoder.layers.8.mlp.fc2.weight', 'model.vision_tower.encoder.layers.8.mlp.fc2.bias', 'model.vision_tower.encoder.layers.8.norm1.weight', 'model.vision_tower.encoder.layers.8.norm1.bias', 'model.vision_tower.encoder.layers.8.norm2.weight', 'model.vision_tower.encoder.layers.8.norm2.bias', 'model.vision_tower.encoder.layers.9.ls1', 'model.vision_tower.encoder.layers.9.ls2', 'model.vision_tower.encoder.layers.9.attn.qkv.weight', 'model.vision_tower.encoder.layers.9.attn.qkv.bias', 'model.vision_tower.encoder.layers.9.attn.proj.weight', 'model.vision_tower.encoder.layers.9.attn.proj.bias', 'model.vision_tower.encoder.layers.9.mlp.fc1.weight', 'model.vision_tower.encoder.layers.9.mlp.fc1.bias', 'model.vision_tower.encoder.layers.9.mlp.fc2.weight', 'model.vision_tower.encoder.layers.9.mlp.fc2.bias', 'model.vision_tower.encoder.layers.9.norm1.weight', 'model.vision_tower.encoder.layers.9.norm1.bias', 'model.vision_tower.encoder.layers.9.norm2.weight', 'model.vision_tower.encoder.layers.9.norm2.bias', 'model.vision_tower.encoder.layers.10.ls1', 'model.vision_tower.encoder.layers.10.ls2', 'model.vision_tower.encoder.layers.10.attn.qkv.weight', 'model.vision_tower.encoder.layers.10.attn.qkv.bias', 'model.vision_tower.encoder.layers.10.attn.proj.weight', 'model.vision_tower.encoder.layers.10.attn.proj.bias', 'model.vision_tower.encoder.layers.10.mlp.fc1.weight', 'model.vision_tower.encoder.layers.10.mlp.fc1.bias', 'model.vision_tower.encoder.layers.10.mlp.fc2.weight', 'model.vision_tower.encoder.layers.10.mlp.fc2.bias', 'model.vision_tower.encoder.layers.10.norm1.weight', 'model.vision_tower.encoder.layers.10.norm1.bias', 'model.vision_tower.encoder.layers.10.norm2.weight', 'model.vision_tower.encoder.layers.10.norm2.bias', 'model.vision_tower.encoder.layers.11.ls1', 'model.vision_tower.encoder.layers.11.ls2', 'model.vision_tower.encoder.layers.11.attn.qkv.weight', 'model.vision_tower.encoder.layers.11.attn.qkv.bias', 'model.vision_tower.encoder.layers.11.attn.proj.weight', 'model.vision_tower.encoder.layers.11.attn.proj.bias', 'model.vision_tower.encoder.layers.11.mlp.fc1.weight', 'model.vision_tower.encoder.layers.11.mlp.fc1.bias', 'model.vision_tower.encoder.layers.11.mlp.fc2.weight', 'model.vision_tower.encoder.layers.11.mlp.fc2.bias', 'model.vision_tower.encoder.layers.11.norm1.weight', 'model.vision_tower.encoder.layers.11.norm1.bias', 'model.vision_tower.encoder.layers.11.norm2.weight', 'model.vision_tower.encoder.layers.11.norm2.bias', 'model.vision_tower.encoder.layers.12.ls1', 'model.vision_tower.encoder.layers.12.ls2', 'model.vision_tower.encoder.layers.12.attn.qkv.weight', 'model.vision_tower.encoder.layers.12.attn.qkv.bias', 'model.vision_tower.encoder.layers.12.attn.proj.weight', 'model.vision_tower.encoder.layers.12.attn.proj.bias', 'model.vision_tower.encoder.layers.12.mlp.fc1.weight', 'model.vision_tower.encoder.layers.12.mlp.fc1.bias', 'model.vision_tower.encoder.layers.12.mlp.fc2.weight', 'model.vision_tower.encoder.layers.12.mlp.fc2.bias', 'model.vision_tower.encoder.layers.12.norm1.weight', 'model.vision_tower.encoder.layers.12.norm1.bias', 'model.vision_tower.encoder.layers.12.norm2.weight', 'model.vision_tower.encoder.layers.12.norm2.bias', 'model.vision_tower.encoder.layers.13.ls1', 'model.vision_tower.encoder.layers.13.ls2', 'model.vision_tower.encoder.layers.13.attn.qkv.weight', 'model.vision_tower.encoder.layers.13.attn.qkv.bias', 'model.vision_tower.encoder.layers.13.attn.proj.weight', 'model.vision_tower.encoder.layers.13.attn.proj.bias', 'model.vision_tower.encoder.layers.13.mlp.fc1.weight', 'model.vision_tower.encoder.layers.13.mlp.fc1.bias', 'model.vision_tower.encoder.layers.13.mlp.fc2.weight', 'model.vision_tower.encoder.layers.13.mlp.fc2.bias', 'model.vision_tower.encoder.layers.13.norm1.weight', 'model.vision_tower.encoder.layers.13.norm1.bias', 'model.vision_tower.encoder.layers.13.norm2.weight', 'model.vision_tower.encoder.layers.13.norm2.bias', 'model.vision_tower.encoder.layers.14.ls1', 'model.vision_tower.encoder.layers.14.ls2', 'model.vision_tower.encoder.layers.14.attn.qkv.weight', 'model.vision_tower.encoder.layers.14.attn.qkv.bias', 'model.vision_tower.encoder.layers.14.attn.proj.weight', 'model.vision_tower.encoder.layers.14.attn.proj.bias', 'model.vision_tower.encoder.layers.14.mlp.fc1.weight', 'model.vision_tower.encoder.layers.14.mlp.fc1.bias', 'model.vision_tower.encoder.layers.14.mlp.fc2.weight', 'model.vision_tower.encoder.layers.14.mlp.fc2.bias', 'model.vision_tower.encoder.layers.14.norm1.weight', 'model.vision_tower.encoder.layers.14.norm1.bias', 'model.vision_tower.encoder.layers.14.norm2.weight', 'model.vision_tower.encoder.layers.14.norm2.bias', 'model.vision_tower.encoder.layers.15.ls1', 'model.vision_tower.encoder.layers.15.ls2', 'model.vision_tower.encoder.layers.15.attn.qkv.weight', 'model.vision_tower.encoder.layers.15.attn.qkv.bias', 'model.vision_tower.encoder.layers.15.attn.proj.weight', 'model.vision_tower.encoder.layers.15.attn.proj.bias', 'model.vision_tower.encoder.layers.15.mlp.fc1.weight', 'model.vision_tower.encoder.layers.15.mlp.fc1.bias', 'model.vision_tower.encoder.layers.15.mlp.fc2.weight', 'model.vision_tower.encoder.layers.15.mlp.fc2.bias', 'model.vision_tower.encoder.layers.15.norm1.weight', 'model.vision_tower.encoder.layers.15.norm1.bias', 'model.vision_tower.encoder.layers.15.norm2.weight', 'model.vision_tower.encoder.layers.15.norm2.bias', 'model.vision_tower.encoder.layers.16.ls1', 'model.vision_tower.encoder.layers.16.ls2', 'model.vision_tower.encoder.layers.16.attn.qkv.weight', 'model.vision_tower.encoder.layers.16.attn.qkv.bias', 'model.vision_tower.encoder.layers.16.attn.proj.weight', 'model.vision_tower.encoder.layers.16.attn.proj.bias', 'model.vision_tower.encoder.layers.16.mlp.fc1.weight', 'model.vision_tower.encoder.layers.16.mlp.fc1.bias', 'model.vision_tower.encoder.layers.16.mlp.fc2.weight', 'model.vision_tower.encoder.layers.16.mlp.fc2.bias', 'model.vision_tower.encoder.layers.16.norm1.weight', 'model.vision_tower.encoder.layers.16.norm1.bias', 'model.vision_tower.encoder.layers.16.norm2.weight', 'model.vision_tower.encoder.layers.16.norm2.bias', 'model.vision_tower.encoder.layers.17.ls1', 'model.vision_tower.encoder.layers.17.ls2', 'model.vision_tower.encoder.layers.17.attn.qkv.weight', 'model.vision_tower.encoder.layers.17.attn.qkv.bias', 'model.vision_tower.encoder.layers.17.attn.proj.weight', 'model.vision_tower.encoder.layers.17.attn.proj.bias', 'model.vision_tower.encoder.layers.17.mlp.fc1.weight', 'model.vision_tower.encoder.layers.17.mlp.fc1.bias', 'model.vision_tower.encoder.layers.17.mlp.fc2.weight', 'model.vision_tower.encoder.layers.17.mlp.fc2.bias', 'model.vision_tower.encoder.layers.17.norm1.weight', 'model.vision_tower.encoder.layers.17.norm1.bias', 'model.vision_tower.encoder.layers.17.norm2.weight', 'model.vision_tower.encoder.layers.17.norm2.bias', 'model.vision_tower.encoder.layers.18.ls1', 'model.vision_tower.encoder.layers.18.ls2', 'model.vision_tower.encoder.layers.18.attn.qkv.weight', 'model.vision_tower.encoder.layers.18.attn.qkv.bias', 'model.vision_tower.encoder.layers.18.attn.proj.weight', 'model.vision_tower.encoder.layers.18.attn.proj.bias', 'model.vision_tower.encoder.layers.18.mlp.fc1.weight', 'model.vision_tower.encoder.layers.18.mlp.fc1.bias', 'model.vision_tower.encoder.layers.18.mlp.fc2.weight', 'model.vision_tower.encoder.layers.18.mlp.fc2.bias', 'model.vision_tower.encoder.layers.18.norm1.weight', 'model.vision_tower.encoder.layers.18.norm1.bias', 'model.vision_tower.encoder.layers.18.norm2.weight', 'model.vision_tower.encoder.layers.18.norm2.bias', 'model.vision_tower.encoder.layers.19.ls1', 'model.vision_tower.encoder.layers.19.ls2', 'model.vision_tower.encoder.layers.19.attn.qkv.weight', 'model.vision_tower.encoder.layers.19.attn.qkv.bias', 'model.vision_tower.encoder.layers.19.attn.proj.weight', 'model.vision_tower.encoder.layers.19.attn.proj.bias', 'model.vision_tower.encoder.layers.19.mlp.fc1.weight', 'model.vision_tower.encoder.layers.19.mlp.fc1.bias', 'model.vision_tower.encoder.layers.19.mlp.fc2.weight', 'model.vision_tower.encoder.layers.19.mlp.fc2.bias', 'model.vision_tower.encoder.layers.19.norm1.weight', 'model.vision_tower.encoder.layers.19.norm1.bias', 'model.vision_tower.encoder.layers.19.norm2.weight', 'model.vision_tower.encoder.layers.19.norm2.bias', 'model.vision_tower.encoder.layers.20.ls1', 'model.vision_tower.encoder.layers.20.ls2', 'model.vision_tower.encoder.layers.20.attn.qkv.weight', 'model.vision_tower.encoder.layers.20.attn.qkv.bias', 'model.vision_tower.encoder.layers.20.attn.proj.weight', 'model.vision_tower.encoder.layers.20.attn.proj.bias', 'model.vision_tower.encoder.layers.20.mlp.fc1.weight', 'model.vision_tower.encoder.layers.20.mlp.fc1.bias', 'model.vision_tower.encoder.layers.20.mlp.fc2.weight', 'model.vision_tower.encoder.layers.20.mlp.fc2.bias', 'model.vision_tower.encoder.layers.20.norm1.weight', 'model.vision_tower.encoder.layers.20.norm1.bias', 'model.vision_tower.encoder.layers.20.norm2.weight', 'model.vision_tower.encoder.layers.20.norm2.bias', 'model.vision_tower.encoder.layers.21.ls1', 'model.vision_tower.encoder.layers.21.ls2', 'model.vision_tower.encoder.layers.21.attn.qkv.weight', 'model.vision_tower.encoder.layers.21.attn.qkv.bias', 'model.vision_tower.encoder.layers.21.attn.proj.weight', 'model.vision_tower.encoder.layers.21.attn.proj.bias', 'model.vision_tower.encoder.layers.21.mlp.fc1.weight', 'model.vision_tower.encoder.layers.21.mlp.fc1.bias', 'model.vision_tower.encoder.layers.21.mlp.fc2.weight', 'model.vision_tower.encoder.layers.21.mlp.fc2.bias', 'model.vision_tower.encoder.layers.21.norm1.weight', 'model.vision_tower.encoder.layers.21.norm1.bias', 'model.vision_tower.encoder.layers.21.norm2.weight', 'model.vision_tower.encoder.layers.21.norm2.bias', 'model.vision_tower.encoder.layers.22.ls1', 'model.vision_tower.encoder.layers.22.ls2', 'model.vision_tower.encoder.layers.22.attn.qkv.weight', 'model.vision_tower.encoder.layers.22.attn.qkv.bias', 'model.vision_tower.encoder.layers.22.attn.proj.weight', 'model.vision_tower.encoder.layers.22.attn.proj.bias', 'model.vision_tower.encoder.layers.22.mlp.fc1.weight', 'model.vision_tower.encoder.layers.22.mlp.fc1.bias', 'model.vision_tower.encoder.layers.22.mlp.fc2.weight', 'model.vision_tower.encoder.layers.22.mlp.fc2.bias', 'model.vision_tower.encoder.layers.22.norm1.weight', 'model.vision_tower.encoder.layers.22.norm1.bias', 'model.vision_tower.encoder.layers.22.norm2.weight', 'model.vision_tower.encoder.layers.22.norm2.bias', 'model.vision_tower.encoder.layers.23.ls1', 'model.vision_tower.encoder.layers.23.ls2', 'model.vision_tower.encoder.layers.23.attn.qkv.weight', 'model.vision_tower.encoder.layers.23.attn.qkv.bias', 'model.vision_tower.encoder.layers.23.attn.proj.weight', 'model.vision_tower.encoder.layers.23.attn.proj.bias', 'model.vision_tower.encoder.layers.23.mlp.fc1.weight', 'model.vision_tower.encoder.layers.23.mlp.fc1.bias', 'model.vision_tower.encoder.layers.23.mlp.fc2.weight', 'model.vision_tower.encoder.layers.23.mlp.fc2.bias', 'model.vision_tower.encoder.layers.23.norm1.weight', 'model.vision_tower.encoder.layers.23.norm1.bias', 'model.vision_tower.encoder.layers.23.norm2.weight', 'model.vision_tower.encoder.layers.23.norm2.bias',
## 'model.multi_modal_projector
'model.multi_modal_projector.0.weight', 'model.multi_modal_projector.0.bias', 'model.multi_modal_projector.1.weight', 'model.multi_modal_projector.1.bias', 'model.multi_modal_projector.3.weight', 'model.multi_modal_projector.3.bias',
## 'model.language_model
'model.language_model.embed_tokens.weight', 'model.language_model.layers.0.self_attn.q_proj.weight', 'model.language_model.layers.0.self_attn.q_proj.bias', 'model.language_model.layers.0.self_attn.k_proj.weight', 'model.language_model.layers.0.self_attn.k_proj.bias', 'model.language_model.layers.0.self_attn.v_proj.weight', 'model.language_model.layers.0.self_attn.v_proj.bias', 'model.language_model.layers.0.self_attn.o_proj.weight', 'model.language_model.layers.0.mlp.gate_proj.weight', 'model.language_model.layers.0.mlp.up_proj.weight', 'model.language_model.layers.0.mlp.down_proj.weight', 'model.language_model.layers.0.input_layernorm.weight', 'model.language_model.layers.0.post_attention_layernorm.weight', 'model.language_model.layers.1.self_attn.q_proj.weight', 'model.language_model.layers.1.self_attn.q_proj.bias', 'model.language_model.layers.1.self_attn.k_proj.weight', 'model.language_model.layers.1.self_attn.k_proj.bias', 'model.language_model.layers.1.self_attn.v_proj.weight', 'model.language_model.layers.1.self_attn.v_proj.bias', 'model.language_model.layers.1.self_attn.o_proj.weight', 'model.language_model.layers.1.mlp.gate_proj.weight', 'model.language_model.layers.1.mlp.up_proj.weight', 'model.language_model.layers.1.mlp.down_proj.weight', 'model.language_model.layers.1.input_layernorm.weight', 'model.language_model.layers.1.post_attention_layernorm.weight', 'model.language_model.layers.2.self_attn.q_proj.weight', 'model.language_model.layers.2.self_attn.q_proj.bias', 'model.language_model.layers.2.self_attn.k_proj.weight', 'model.language_model.layers.2.self_attn.k_proj.bias', 'model.language_model.layers.2.self_attn.v_proj.weight', 'model.language_model.layers.2.self_attn.v_proj.bias', 'model.language_model.layers.2.self_attn.o_proj.weight', 'model.language_model.layers.2.mlp.gate_proj.weight', 'model.language_model.layers.2.mlp.up_proj.weight', 'model.language_model.layers.2.mlp.down_proj.weight', 'model.language_model.layers.2.input_layernorm.weight', 'model.language_model.layers.2.post_attention_layernorm.weight', 'model.language_model.layers.3.self_attn.q_proj.weight', 'model.language_model.layers.3.self_attn.q_proj.bias', 'model.language_model.layers.3.self_attn.k_proj.weight', 'model.language_model.layers.3.self_attn.k_proj.bias', 'model.language_model.layers.3.self_attn.v_proj.weight', 'model.language_model.layers.3.self_attn.v_proj.bias', 'model.language_model.layers.3.self_attn.o_proj.weight', 'model.language_model.layers.3.mlp.gate_proj.weight', 'model.language_model.layers.3.mlp.up_proj.weight', 'model.language_model.layers.3.mlp.down_proj.weight', 'model.language_model.layers.3.input_layernorm.weight', 'model.language_model.layers.3.post_attention_layernorm.weight', 'model.language_model.layers.4.self_attn.q_proj.weight', 'model.language_model.layers.4.self_attn.q_proj.bias', 'model.language_model.layers.4.self_attn.k_proj.weight', 'model.language_model.layers.4.self_attn.k_proj.bias', 'model.language_model.layers.4.self_attn.v_proj.weight', 'model.language_model.layers.4.self_attn.v_proj.bias', 'model.language_model.layers.4.self_attn.o_proj.weight', 'model.language_model.layers.4.mlp.gate_proj.weight', 'model.language_model.layers.4.mlp.up_proj.weight', 'model.language_model.layers.4.mlp.down_proj.weight', 'model.language_model.layers.4.input_layernorm.weight', 'model.language_model.layers.4.post_attention_layernorm.weight', 'model.language_model.layers.5.self_attn.q_proj.weight', 'model.language_model.layers.5.self_attn.q_proj.bias', 'model.language_model.layers.5.self_attn.k_proj.weight', 'model.language_model.layers.5.self_attn.k_proj.bias', 'model.language_model.layers.5.self_attn.v_proj.weight', 'model.language_model.layers.5.self_attn.v_proj.bias', 'model.language_model.layers.5.self_attn.o_proj.weight', 'model.language_model.layers.5.mlp.gate_proj.weight', 'model.language_model.layers.5.mlp.up_proj.weight', 'model.language_model.layers.5.mlp.down_proj.weight', 'model.language_model.layers.5.input_layernorm.weight', 'model.language_model.layers.5.post_attention_layernorm.weight', 'model.language_model.layers.6.self_attn.q_proj.weight', 'model.language_model.layers.6.self_attn.q_proj.bias', 'model.language_model.layers.6.self_attn.k_proj.weight', 'model.language_model.layers.6.self_attn.k_proj.bias', 'model.language_model.layers.6.self_attn.v_proj.weight', 'model.language_model.layers.6.self_attn.v_proj.bias', 'model.language_model.layers.6.self_attn.o_proj.weight', 'model.language_model.layers.6.mlp.gate_proj.weight', 'model.language_model.layers.6.mlp.up_proj.weight', 'model.language_model.layers.6.mlp.down_proj.weight', 'model.language_model.layers.6.input_layernorm.weight', 'model.language_model.layers.6.post_attention_layernorm.weight', 'model.language_model.layers.7.self_attn.q_proj.weight', 'model.language_model.layers.7.self_attn.q_proj.bias', 'model.language_model.layers.7.self_attn.k_proj.weight', 'model.language_model.layers.7.self_attn.k_proj.bias', 'model.language_model.layers.7.self_attn.v_proj.weight', 'model.language_model.layers.7.self_attn.v_proj.bias', 'model.language_model.layers.7.self_attn.o_proj.weight', 'model.language_model.layers.7.mlp.gate_proj.weight', 'model.language_model.layers.7.mlp.up_proj.weight', 'model.language_model.layers.7.mlp.down_proj.weight', 'model.language_model.layers.7.input_layernorm.weight', 'model.language_model.layers.7.post_attention_layernorm.weight', 'model.language_model.layers.8.self_attn.q_proj.weight', 'model.language_model.layers.8.self_attn.q_proj.bias', 'model.language_model.layers.8.self_attn.k_proj.weight', 'model.language_model.layers.8.self_attn.k_proj.bias', 'model.language_model.layers.8.self_attn.v_proj.weight', 'model.language_model.layers.8.self_attn.v_proj.bias', 'model.language_model.layers.8.self_attn.o_proj.weight', 'model.language_model.layers.8.mlp.gate_proj.weight', 'model.language_model.layers.8.mlp.up_proj.weight', 'model.language_model.layers.8.mlp.down_proj.weight', 'model.language_model.layers.8.input_layernorm.weight', 'model.language_model.layers.8.post_attention_layernorm.weight', 'model.language_model.layers.9.self_attn.q_proj.weight', 'model.language_model.layers.9.self_attn.q_proj.bias', 'model.language_model.layers.9.self_attn.k_proj.weight', 'model.language_model.layers.9.self_attn.k_proj.bias', 'model.language_model.layers.9.self_attn.v_proj.weight', 'model.language_model.layers.9.self_attn.v_proj.bias', 'model.language_model.layers.9.self_attn.o_proj.weight', 'model.language_model.layers.9.mlp.gate_proj.weight', 'model.language_model.layers.9.mlp.up_proj.weight', 'model.language_model.layers.9.mlp.down_proj.weight', 'model.language_model.layers.9.input_layernorm.weight', 'model.language_model.layers.9.post_attention_layernorm.weight', 'model.language_model.layers.10.self_attn.q_proj.weight', 'model.language_model.layers.10.self_attn.q_proj.bias', 'model.language_model.layers.10.self_attn.k_proj.weight', 'model.language_model.layers.10.self_attn.k_proj.bias', 'model.language_model.layers.10.self_attn.v_proj.weight', 'model.language_model.layers.10.self_attn.v_proj.bias', 'model.language_model.layers.10.self_attn.o_proj.weight', 'model.language_model.layers.10.mlp.gate_proj.weight', 'model.language_model.layers.10.mlp.up_proj.weight', 'model.language_model.layers.10.mlp.down_proj.weight', 'model.language_model.layers.10.input_layernorm.weight', 'model.language_model.layers.10.post_attention_layernorm.weight', 'model.language_model.layers.11.self_attn.q_proj.weight', 'model.language_model.layers.11.self_attn.q_proj.bias', 'model.language_model.layers.11.self_attn.k_proj.weight', 'model.language_model.layers.11.self_attn.k_proj.bias', 'model.language_model.layers.11.self_attn.v_proj.weight', 'model.language_model.layers.11.self_attn.v_proj.bias', 'model.language_model.layers.11.self_attn.o_proj.weight', 'model.language_model.layers.11.mlp.gate_proj.weight', 'model.language_model.layers.11.mlp.up_proj.weight', 'model.language_model.layers.11.mlp.down_proj.weight', 'model.language_model.layers.11.input_layernorm.weight', 'model.language_model.layers.11.post_attention_layernorm.weight', 'model.language_model.layers.12.self_attn.q_proj.weight', 'model.language_model.layers.12.self_attn.q_proj.bias', 'model.language_model.layers.12.self_attn.k_proj.weight', 'model.language_model.layers.12.self_attn.k_proj.bias', 'model.language_model.layers.12.self_attn.v_proj.weight', 'model.language_model.layers.12.self_attn.v_proj.bias', 'model.language_model.layers.12.self_attn.o_proj.weight', 'model.language_model.layers.12.mlp.gate_proj.weight', 'model.language_model.layers.12.mlp.up_proj.weight', 'model.language_model.layers.12.mlp.down_proj.weight', 'model.language_model.layers.12.input_layernorm.weight', 'model.language_model.layers.12.post_attention_layernorm.weight', 'model.language_model.layers.13.self_attn.q_proj.weight', 'model.language_model.layers.13.self_attn.q_proj.bias', 'model.language_model.layers.13.self_attn.k_proj.weight', 'model.language_model.layers.13.self_attn.k_proj.bias', 'model.language_model.layers.13.self_attn.v_proj.weight', 'model.language_model.layers.13.self_attn.v_proj.bias', 'model.language_model.layers.13.self_attn.o_proj.weight', 'model.language_model.layers.13.mlp.gate_proj.weight', 'model.language_model.layers.13.mlp.up_proj.weight', 'model.language_model.layers.13.mlp.down_proj.weight', 'model.language_model.layers.13.input_layernorm.weight', 'model.language_model.layers.13.post_attention_layernorm.weight', 'model.language_model.layers.14.self_attn.q_proj.weight', 'model.language_model.layers.14.self_attn.q_proj.bias', 'model.language_model.layers.14.self_attn.k_proj.weight', 'model.language_model.layers.14.self_attn.k_proj.bias', 'model.language_model.layers.14.self_attn.v_proj.weight', 'model.language_model.layers.14.self_attn.v_proj.bias', 'model.language_model.layers.14.self_attn.o_proj.weight', 'model.language_model.layers.14.mlp.gate_proj.weight', 'model.language_model.layers.14.mlp.up_proj.weight', 'model.language_model.layers.14.mlp.down_proj.weight', 'model.language_model.layers.14.input_layernorm.weight', 'model.language_model.layers.14.post_attention_layernorm.weight', 'model.language_model.layers.15.self_attn.q_proj.weight', 'model.language_model.layers.15.self_attn.q_proj.bias', 'model.language_model.layers.15.self_attn.k_proj.weight', 'model.language_model.layers.15.self_attn.k_proj.bias', 'model.language_model.layers.15.self_attn.v_proj.weight', 'model.language_model.layers.15.self_attn.v_proj.bias', 'model.language_model.layers.15.self_attn.o_proj.weight', 'model.language_model.layers.15.mlp.gate_proj.weight', 'model.language_model.layers.15.mlp.up_proj.weight', 'model.language_model.layers.15.mlp.down_proj.weight', 'model.language_model.layers.15.input_layernorm.weight', 'model.language_model.layers.15.post_attention_layernorm.weight', 'model.language_model.layers.16.self_attn.q_proj.weight', 'model.language_model.layers.16.self_attn.q_proj.bias', 'model.language_model.layers.16.self_attn.k_proj.weight', 'model.language_model.layers.16.self_attn.k_proj.bias', 'model.language_model.layers.16.self_attn.v_proj.weight', 'model.language_model.layers.16.self_attn.v_proj.bias', 'model.language_model.layers.16.self_attn.o_proj.weight', 'model.language_model.layers.16.mlp.gate_proj.weight', 'model.language_model.layers.16.mlp.up_proj.weight', 'model.language_model.layers.16.mlp.down_proj.weight', 'model.language_model.layers.16.input_layernorm.weight', 'model.language_model.layers.16.post_attention_layernorm.weight', 'model.language_model.layers.17.self_attn.q_proj.weight', 'model.language_model.layers.17.self_attn.q_proj.bias', 'model.language_model.layers.17.self_attn.k_proj.weight', 'model.language_model.layers.17.self_attn.k_proj.bias', 'model.language_model.layers.17.self_attn.v_proj.weight', 'model.language_model.layers.17.self_attn.v_proj.bias', 'model.language_model.layers.17.self_attn.o_proj.weight', 'model.language_model.layers.17.mlp.gate_proj.weight', 'model.language_model.layers.17.mlp.up_proj.weight', 'model.language_model.layers.17.mlp.down_proj.weight', 'model.language_model.layers.17.input_layernorm.weight', 'model.language_model.layers.17.post_attention_layernorm.weight', 'model.language_model.layers.18.self_attn.q_proj.weight', 'model.language_model.layers.18.self_attn.q_proj.bias', 'model.language_model.layers.18.self_attn.k_proj.weight', 'model.language_model.layers.18.self_attn.k_proj.bias', 'model.language_model.layers.18.self_attn.v_proj.weight', 'model.language_model.layers.18.self_attn.v_proj.bias', 'model.language_model.layers.18.self_attn.o_proj.weight', 'model.language_model.layers.18.mlp.gate_proj.weight', 'model.language_model.layers.18.mlp.up_proj.weight', 'model.language_model.layers.18.mlp.down_proj.weight', 'model.language_model.layers.18.input_layernorm.weight', 'model.language_model.layers.18.post_attention_layernorm.weight', 'model.language_model.layers.19.self_attn.q_proj.weight', 'model.language_model.layers.19.self_attn.q_proj.bias', 'model.language_model.layers.19.self_attn.k_proj.weight', 'model.language_model.layers.19.self_attn.k_proj.bias', 'model.language_model.layers.19.self_attn.v_proj.weight', 'model.language_model.layers.19.self_attn.v_proj.bias', 'model.language_model.layers.19.self_attn.o_proj.weight', 'model.language_model.layers.19.mlp.gate_proj.weight', 'model.language_model.layers.19.mlp.up_proj.weight', 'model.language_model.layers.19.mlp.down_proj.weight', 'model.language_model.layers.19.input_layernorm.weight', 'model.language_model.layers.19.post_attention_layernorm.weight', 'model.language_model.layers.20.self_attn.q_proj.weight', 'model.language_model.layers.20.self_attn.q_proj.bias', 'model.language_model.layers.20.self_attn.k_proj.weight', 'model.language_model.layers.20.self_attn.k_proj.bias', 'model.language_model.layers.20.self_attn.v_proj.weight', 'model.language_model.layers.20.self_attn.v_proj.bias', 'model.language_model.layers.20.self_attn.o_proj.weight', 'model.language_model.layers.20.mlp.gate_proj.weight', 'model.language_model.layers.20.mlp.up_proj.weight', 'model.language_model.layers.20.mlp.down_proj.weight', 'model.language_model.layers.20.input_layernorm.weight', 'model.language_model.layers.20.post_attention_layernorm.weight', 'model.language_model.layers.21.self_attn.q_proj.weight', 'model.language_model.layers.21.self_attn.q_proj.bias', 'model.language_model.layers.21.self_attn.k_proj.weight', 'model.language_model.layers.21.self_attn.k_proj.bias', 'model.language_model.layers.21.self_attn.v_proj.weight', 'model.language_model.layers.21.self_attn.v_proj.bias', 'model.language_model.layers.21.self_attn.o_proj.weight', 'model.language_model.layers.21.mlp.gate_proj.weight', 'model.language_model.layers.21.mlp.up_proj.weight', 'model.language_model.layers.21.mlp.down_proj.weight', 'model.language_model.layers.21.input_layernorm.weight', 'model.language_model.layers.21.post_attention_layernorm.weight', 'model.language_model.layers.22.self_attn.q_proj.weight', 'model.language_model.layers.22.self_attn.q_proj.bias', 'model.language_model.layers.22.self_attn.k_proj.weight', 'model.language_model.layers.22.self_attn.k_proj.bias', 'model.language_model.layers.22.self_attn.v_proj.weight', 'model.language_model.layers.22.self_attn.v_proj.bias', 'model.language_model.layers.22.self_attn.o_proj.weight', 'model.language_model.layers.22.mlp.gate_proj.weight', 'model.language_model.layers.22.mlp.up_proj.weight', 'model.language_model.layers.22.mlp.down_proj.weight', 'model.language_model.layers.22.input_layernorm.weight', 'model.language_model.layers.22.post_attention_layernorm.weight', 'model.language_model.layers.23.self_attn.q_proj.weight', 'model.language_model.layers.23.self_attn.q_proj.bias', 'model.language_model.layers.23.self_attn.k_proj.weight', 'model.language_model.layers.23.self_attn.k_proj.bias', 'model.language_model.layers.23.self_attn.v_proj.weight', 'model.language_model.layers.23.self_attn.v_proj.bias', 'model.language_model.layers.23.self_attn.o_proj.weight', 'model.language_model.layers.23.mlp.gate_proj.weight', 'model.language_model.layers.23.mlp.up_proj.weight', 'model.language_model.layers.23.mlp.down_proj.weight', 'model.language_model.layers.23.input_layernorm.weight', 'model.language_model.layers.23.post_attention_layernorm.weight', 'model.language_model.norm.weight',
## 'model.dit
'model.dit.scale_shift_table', 'model.dit.patch_embed.proj.weight', 'model.dit.patch_embed.proj.bias', 'model.dit.time_embed.emb.timestep_embedder.linear_1.weight', 'model.dit.time_embed.emb.timestep_embedder.linear_1.bias', 'model.dit.time_embed.emb.timestep_embedder.linear_2.weight', 'model.dit.time_embed.emb.timestep_embedder.linear_2.bias', 'model.dit.time_embed.linear.weight', 'model.dit.time_embed.linear.bias', 'model.dit.caption_projection.linear_1.weight', 'model.dit.caption_projection.linear_1.bias', 'model.dit.caption_projection.linear_2.weight', 'model.dit.caption_projection.linear_2.bias', 'model.dit.caption_norm.weight', 'model.dit.transformer_blocks.0.scale_shift_table', 'model.dit.transformer_blocks.0.attn1.to_q.weight', 'model.dit.transformer_blocks.0.attn1.to_k.weight', 'model.dit.transformer_blocks.0.attn1.to_v.weight', 'model.dit.transformer_blocks.0.attn1.to_out.0.weight', 'model.dit.transformer_blocks.0.attn1.to_out.0.bias', 'model.dit.transformer_blocks.0.attn2.to_q.weight', 'model.dit.transformer_blocks.0.attn2.to_q.bias', 'model.dit.transformer_blocks.0.attn2.to_k.weight', 'model.dit.transformer_blocks.0.attn2.to_k.bias', 'model.dit.transformer_blocks.0.attn2.to_v.weight', 'model.dit.transformer_blocks.0.attn2.to_v.bias', 'model.dit.transformer_blocks.0.attn2.to_out.0.weight', 'model.dit.transformer_blocks.0.attn2.to_out.0.bias', 'model.dit.transformer_blocks.0.ff.conv_inverted.weight', 'model.dit.transformer_blocks.0.ff.conv_inverted.bias', 'model.dit.transformer_blocks.0.ff.conv_depth.weight', 'model.dit.transformer_blocks.0.ff.conv_depth.bias', 'model.dit.transformer_blocks.0.ff.conv_point.weight', 'model.dit.transformer_blocks.1.scale_shift_table', 'model.dit.transformer_blocks.1.attn1.to_q.weight', 'model.dit.transformer_blocks.1.attn1.to_k.weight', 'model.dit.transformer_blocks.1.attn1.to_v.weight', 'model.dit.transformer_blocks.1.attn1.to_out.0.weight', 'model.dit.transformer_blocks.1.attn1.to_out.0.bias', 'model.dit.transformer_blocks.1.attn2.to_q.weight', 'model.dit.transformer_blocks.1.attn2.to_q.bias', 'model.dit.transformer_blocks.1.attn2.to_k.weight', 'model.dit.transformer_blocks.1.attn2.to_k.bias', 'model.dit.transformer_blocks.1.attn2.to_v.weight', 'model.dit.transformer_blocks.1.attn2.to_v.bias', 'model.dit.transformer_blocks.1.attn2.to_out.0.weight', 'model.dit.transformer_blocks.1.attn2.to_out.0.bias', 'model.dit.transformer_blocks.1.ff.conv_inverted.weight', 'model.dit.transformer_blocks.1.ff.conv_inverted.bias', 'model.dit.transformer_blocks.1.ff.conv_depth.weight', 'model.dit.transformer_blocks.1.ff.conv_depth.bias', 'model.dit.transformer_blocks.1.ff.conv_point.weight', 'model.dit.transformer_blocks.2.scale_shift_table', 'model.dit.transformer_blocks.2.attn1.to_q.weight', 'model.dit.transformer_blocks.2.attn1.to_k.weight', 'model.dit.transformer_blocks.2.attn1.to_v.weight', 'model.dit.transformer_blocks.2.attn1.to_out.0.weight', 'model.dit.transformer_blocks.2.attn1.to_out.0.bias', 'model.dit.transformer_blocks.2.attn2.to_q.weight', 'model.dit.transformer_blocks.2.attn2.to_q.bias', 'model.dit.transformer_blocks.2.attn2.to_k.weight', 'model.dit.transformer_blocks.2.attn2.to_k.bias', 'model.dit.transformer_blocks.2.attn2.to_v.weight', 'model.dit.transformer_blocks.2.attn2.to_v.bias', 'model.dit.transformer_blocks.2.attn2.to_out.0.weight', 'model.dit.transformer_blocks.2.attn2.to_out.0.bias', 'model.dit.transformer_blocks.2.ff.conv_inverted.weight', 'model.dit.transformer_blocks.2.ff.conv_inverted.bias', 'model.dit.transformer_blocks.2.ff.conv_depth.weight', 'model.dit.transformer_blocks.2.ff.conv_depth.bias', 'model.dit.transformer_blocks.2.ff.conv_point.weight', 'model.dit.transformer_blocks.3.scale_shift_table', 'model.dit.transformer_blocks.3.attn1.to_q.weight', 'model.dit.transformer_blocks.3.attn1.to_k.weight', 'model.dit.transformer_blocks.3.attn1.to_v.weight', 'model.dit.transformer_blocks.3.attn1.to_out.0.weight', 'model.dit.transformer_blocks.3.attn1.to_out.0.bias', 'model.dit.transformer_blocks.3.attn2.to_q.weight', 'model.dit.transformer_blocks.3.attn2.to_q.bias', 'model.dit.transformer_blocks.3.attn2.to_k.weight', 'model.dit.transformer_blocks.3.attn2.to_k.bias', 'model.dit.transformer_blocks.3.attn2.to_v.weight', 'model.dit.transformer_blocks.3.attn2.to_v.bias', 'model.dit.transformer_blocks.3.attn2.to_out.0.weight', 'model.dit.transformer_blocks.3.attn2.to_out.0.bias', 'model.dit.transformer_blocks.3.ff.conv_inverted.weight', 'model.dit.transformer_blocks.3.ff.conv_inverted.bias', 'model.dit.transformer_blocks.3.ff.conv_depth.weight', 'model.dit.transformer_blocks.3.ff.conv_depth.bias', 'model.dit.transformer_blocks.3.ff.conv_point.weight', 'model.dit.transformer_blocks.4.scale_shift_table', 'model.dit.transformer_blocks.4.attn1.to_q.weight', 'model.dit.transformer_blocks.4.attn1.to_k.weight', 'model.dit.transformer_blocks.4.attn1.to_v.weight', 'model.dit.transformer_blocks.4.attn1.to_out.0.weight', 'model.dit.transformer_blocks.4.attn1.to_out.0.bias', 'model.dit.transformer_blocks.4.attn2.to_q.weight', 'model.dit.transformer_blocks.4.attn2.to_q.bias', 'model.dit.transformer_blocks.4.attn2.to_k.weight', 'model.dit.transformer_blocks.4.attn2.to_k.bias', 'model.dit.transformer_blocks.4.attn2.to_v.weight', 'model.dit.transformer_blocks.4.attn2.to_v.bias', 'model.dit.transformer_blocks.4.attn2.to_out.0.weight', 'model.dit.transformer_blocks.4.attn2.to_out.0.bias', 'model.dit.transformer_blocks.4.ff.conv_inverted.weight', 'model.dit.transformer_blocks.4.ff.conv_inverted.bias', 'model.dit.transformer_blocks.4.ff.conv_depth.weight', 'model.dit.transformer_blocks.4.ff.conv_depth.bias', 'model.dit.transformer_blocks.4.ff.conv_point.weight', 'model.dit.transformer_blocks.5.scale_shift_table', 'model.dit.transformer_blocks.5.attn1.to_q.weight', 'model.dit.transformer_blocks.5.attn1.to_k.weight', 'model.dit.transformer_blocks.5.attn1.to_v.weight', 'model.dit.transformer_blocks.5.attn1.to_out.0.weight', 'model.dit.transformer_blocks.5.attn1.to_out.0.bias', 'model.dit.transformer_blocks.5.attn2.to_q.weight', 'model.dit.transformer_blocks.5.attn2.to_q.bias', 'model.dit.transformer_blocks.5.attn2.to_k.weight', 'model.dit.transformer_blocks.5.attn2.to_k.bias', 'model.dit.transformer_blocks.5.attn2.to_v.weight', 'model.dit.transformer_blocks.5.attn2.to_v.bias', 'model.dit.transformer_blocks.5.attn2.to_out.0.weight', 'model.dit.transformer_blocks.5.attn2.to_out.0.bias', 'model.dit.transformer_blocks.5.ff.conv_inverted.weight', 'model.dit.transformer_blocks.5.ff.conv_inverted.bias', 'model.dit.transformer_blocks.5.ff.conv_depth.weight', 'model.dit.transformer_blocks.5.ff.conv_depth.bias', 'model.dit.transformer_blocks.5.ff.conv_point.weight', 'model.dit.transformer_blocks.6.scale_shift_table', 'model.dit.transformer_blocks.6.attn1.to_q.weight', 'model.dit.transformer_blocks.6.attn1.to_k.weight', 'model.dit.transformer_blocks.6.attn1.to_v.weight', 'model.dit.transformer_blocks.6.attn1.to_out.0.weight', 'model.dit.transformer_blocks.6.attn1.to_out.0.bias', 'model.dit.transformer_blocks.6.attn2.to_q.weight', 'model.dit.transformer_blocks.6.attn2.to_q.bias', 'model.dit.transformer_blocks.6.attn2.to_k.weight', 'model.dit.transformer_blocks.6.attn2.to_k.bias', 'model.dit.transformer_blocks.6.attn2.to_v.weight', 'model.dit.transformer_blocks.6.attn2.to_v.bias', 'model.dit.transformer_blocks.6.attn2.to_out.0.weight', 'model.dit.transformer_blocks.6.attn2.to_out.0.bias', 'model.dit.transformer_blocks.6.ff.conv_inverted.weight', 'model.dit.transformer_blocks.6.ff.conv_inverted.bias', 'model.dit.transformer_blocks.6.ff.conv_depth.weight', 'model.dit.transformer_blocks.6.ff.conv_depth.bias', 'model.dit.transformer_blocks.6.ff.conv_point.weight', 'model.dit.transformer_blocks.7.scale_shift_table', 'model.dit.transformer_blocks.7.attn1.to_q.weight', 'model.dit.transformer_blocks.7.attn1.to_k.weight', 'model.dit.transformer_blocks.7.attn1.to_v.weight', 'model.dit.transformer_blocks.7.attn1.to_out.0.weight', 'model.dit.transformer_blocks.7.attn1.to_out.0.bias', 'model.dit.transformer_blocks.7.attn2.to_q.weight', 'model.dit.transformer_blocks.7.attn2.to_q.bias', 'model.dit.transformer_blocks.7.attn2.to_k.weight', 'model.dit.transformer_blocks.7.attn2.to_k.bias', 'model.dit.transformer_blocks.7.attn2.to_v.weight', 'model.dit.transformer_blocks.7.attn2.to_v.bias', 'model.dit.transformer_blocks.7.attn2.to_out.0.weight', 'model.dit.transformer_blocks.7.attn2.to_out.0.bias', 'model.dit.transformer_blocks.7.ff.conv_inverted.weight', 'model.dit.transformer_blocks.7.ff.conv_inverted.bias', 'model.dit.transformer_blocks.7.ff.conv_depth.weight', 'model.dit.transformer_blocks.7.ff.conv_depth.bias', 'model.dit.transformer_blocks.7.ff.conv_point.weight', 'model.dit.transformer_blocks.8.scale_shift_table', 'model.dit.transformer_blocks.8.attn1.to_q.weight', 'model.dit.transformer_blocks.8.attn1.to_k.weight', 'model.dit.transformer_blocks.8.attn1.to_v.weight', 'model.dit.transformer_blocks.8.attn1.to_out.0.weight', 'model.dit.transformer_blocks.8.attn1.to_out.0.bias', 'model.dit.transformer_blocks.8.attn2.to_q.weight', 'model.dit.transformer_blocks.8.attn2.to_q.bias', 'model.dit.transformer_blocks.8.attn2.to_k.weight', 'model.dit.transformer_blocks.8.attn2.to_k.bias', 'model.dit.transformer_blocks.8.attn2.to_v.weight', 'mode...3.ff.conv_depth.weight', 'model.dit.transformer_blocks.23.ff.conv_depth.bias', 'model.dit.transformer_blocks.23.ff.conv_point.weight', 'model.dit.transformer_blocks.24.scale_shift_table', 'model.dit.transformer_blocks.24.attn1.to_q.weight', 'model.dit.transformer_blocks.24.attn1.to_k.weight', 'model.dit.transformer_blocks.24.attn1.to_v.weight', 'model.dit.transformer_blocks.24.attn1.to_out.0.weight', 'model.dit.transformer_blocks.24.attn1.to_out.0.bias', 'model.dit.transformer_blocks.24.attn2.to_q.weight', 'model.dit.transformer_blocks.24.attn2.to_q.bias', 'model.dit.transformer_blocks.24.attn2.to_k.weight', 'model.dit.transformer_blocks.24.attn2.to_k.bias', 'model.dit.transformer_blocks.24.attn2.to_v.weight', 'model.dit.transformer_blocks.24.attn2.to_v.bias', 'model.dit.transformer_blocks.24.attn2.to_out.0.weight', 'model.dit.transformer_blocks.24.attn2.to_out.0.bias', 'model.dit.transformer_blocks.24.ff.conv_inverted.weight', 'model.dit.transformer_blocks.24.ff.conv_inverted.bias', 'model.dit.transformer_blocks.24.ff.conv_depth.weight', 'model.dit.transformer_blocks.24.ff.conv_depth.bias', 'model.dit.transformer_blocks.24.ff.conv_point.weight', 'model.dit.transformer_blocks.25.scale_shift_table', 'model.dit.transformer_blocks.25.attn1.to_q.weight', 'model.dit.transformer_blocks.25.attn1.to_k.weight', 'model.dit.transformer_blocks.25.attn1.to_v.weight', 'model.dit.transformer_blocks.25.attn1.to_out.0.weight', 'model.dit.transformer_blocks.25.attn1.to_out.0.bias', 'model.dit.transformer_blocks.25.attn2.to_q.weight', 'model.dit.transformer_blocks.25.attn2.to_q.bias', 'model.dit.transformer_blocks.25.attn2.to_k.weight', 'model.dit.transformer_blocks.25.attn2.to_k.bias', 'model.dit.transformer_blocks.25.attn2.to_v.weight', 'model.dit.transformer_blocks.25.attn2.to_v.bias', 'model.dit.transformer_blocks.25.attn2.to_out.0.weight', 'model.dit.transformer_blocks.25.attn2.to_out.0.bias', 'model.dit.transformer_blocks.25.ff.conv_inverted.weight', 'model.dit.transformer_blocks.25.ff.conv_inverted.bias', 'model.dit.transformer_blocks.25.ff.conv_depth.weight', 'model.dit.transformer_blocks.25.ff.conv_depth.bias', 'model.dit.transformer_blocks.25.ff.conv_point.weight', 'model.dit.transformer_blocks.26.scale_shift_table', 'model.dit.transformer_blocks.26.attn1.to_q.weight', 'model.dit.transformer_blocks.26.attn1.to_k.weight', 'model.dit.transformer_blocks.26.attn1.to_v.weight', 'model.dit.transformer_blocks.26.attn1.to_out.0.weight', 'model.dit.transformer_blocks.26.attn1.to_out.0.bias', 'model.dit.transformer_blocks.26.attn2.to_q.weight', 'model.dit.transformer_blocks.26.attn2.to_q.bias', 'model.dit.transformer_blocks.26.attn2.to_k.weight', 'model.dit.transformer_blocks.26.attn2.to_k.bias', 'model.dit.transformer_blocks.26.attn2.to_v.weight', 'model.dit.transformer_blocks.26.attn2.to_v.bias', 'model.dit.transformer_blocks.26.attn2.to_out.0.weight', 'model.dit.transformer_blocks.26.attn2.to_out.0.bias', 'model.dit.transformer_blocks.26.ff.conv_inverted.weight', 'model.dit.transformer_blocks.26.ff.conv_inverted.bias', 'model.dit.transformer_blocks.26.ff.conv_depth.weight', 'model.dit.transformer_blocks.26.ff.conv_depth.bias', 'model.dit.transformer_blocks.26.ff.conv_point.weight', 'model.dit.transformer_blocks.27.scale_shift_table', 'model.dit.transformer_blocks.27.attn1.to_q.weight', 'model.dit.transformer_blocks.27.attn1.to_k.weight', 'model.dit.transformer_blocks.27.attn1.to_v.weight', 'model.dit.transformer_blocks.27.attn1.to_out.0.weight', 'model.dit.transformer_blocks.27.attn1.to_out.0.bias', 'model.dit.transformer_blocks.27.attn2.to_q.weight', 'model.dit.transformer_blocks.27.attn2.to_q.bias', 'model.dit.transformer_blocks.27.attn2.to_k.weight', 'model.dit.transformer_blocks.27.attn2.to_k.bias', 'model.dit.transformer_blocks.27.attn2.to_v.weight', 'model.dit.transformer_blocks.27.attn2.to_v.bias', 'model.dit.transformer_blocks.27.attn2.to_out.0.weight', 'model.dit.transformer_blocks.27.attn2.to_out.0.bias', 'model.dit.transformer_blocks.27.ff.conv_inverted.weight', 'model.dit.transformer_blocks.27.ff.conv_inverted.bias', 'model.dit.transformer_blocks.27.ff.conv_depth.weight', 'model.dit.transformer_blocks.27.ff.conv_depth.bias', 'model.dit.transformer_blocks.27.ff.conv_point.weight', 'model.dit.proj_out.weight', 'model.dit.proj_out.bias',
## 'model.vae_decoder
'model.vae_decoder.decoder.conv_in.weight', 'model.vae_decoder.decoder.conv_in.bias', 'model.vae_decoder.decoder.up_blocks.0.0.conv.weight', 'model.vae_decoder.decoder.up_blocks.0.0.conv.bias', 'model.vae_decoder.decoder.up_blocks.0.1.conv1.weight', 'model.vae_decoder.decoder.up_blocks.0.1.conv1.bias', 'model.vae_decoder.decoder.up_blocks.0.1.conv2.weight', 'model.vae_decoder.decoder.up_blocks.0.1.norm.weight', 'model.vae_decoder.decoder.up_blocks.0.1.norm.bias', 'model.vae_decoder.decoder.up_blocks.0.2.conv1.weight', 'model.vae_decoder.decoder.up_blocks.0.2.conv1.bias', 'model.vae_decoder.decoder.up_blocks.0.2.conv2.weight', 'model.vae_decoder.decoder.up_blocks.0.2.norm.weight', 'model.vae_decoder.decoder.up_blocks.0.2.norm.bias', 'model.vae_decoder.decoder.up_blocks.0.3.conv1.weight', 'model.vae_decoder.decoder.up_blocks.0.3.conv1.bias', 'model.vae_decoder.decoder.up_blocks.0.3.conv2.weight', 'model.vae_decoder.decoder.up_blocks.0.3.norm.weight', 'model.vae_decoder.decoder.up_blocks.0.3.norm.bias', 'model.vae_decoder.decoder.up_blocks.1.0.conv.weight', 'model.vae_decoder.decoder.up_blocks.1.0.conv.bias', 'model.vae_decoder.decoder.up_blocks.1.1.conv1.weight', 'model.vae_decoder.decoder.up_blocks.1.1.conv1.bias', 'model.vae_decoder.decoder.up_blocks.1.1.conv2.weight', 'model.vae_decoder.decoder.up_blocks.1.1.norm.weight', 'model.vae_decoder.decoder.up_blocks.1.1.norm.bias', 'model.vae_decoder.decoder.up_blocks.1.2.conv1.weight', 'model.vae_decoder.decoder.up_blocks.1.2.conv1.bias', 'model.vae_decoder.decoder.up_blocks.1.2.conv2.weight', 'model.vae_decoder.decoder.up_blocks.1.2.norm.weight', 'model.vae_decoder.decoder.up_blocks.1.2.norm.bias', 'model.vae_decoder.decoder.up_blocks.1.3.conv1.weight', 'model.vae_decoder.decoder.up_blocks.1.3.conv1.bias', 'model.vae_decoder.decoder.up_blocks.1.3.conv2.weight', 'model.vae_decoder.decoder.up_blocks.1.3.norm.weight', 'model.vae_decoder.decoder.up_blocks.1.3.norm.bias', 'model.vae_decoder.decoder.up_blocks.2.0.conv.weight', 'model.vae_decoder.decoder.up_blocks.2.0.conv.bias', 'model.vae_decoder.decoder.up_blocks.2.1.conv1.weight', 'model.vae_decoder.decoder.up_blocks.2.1.conv1.bias', 'model.vae_decoder.decoder.up_blocks.2.1.conv2.weight', 'model.vae_decoder.decoder.up_blocks.2.1.norm.weight', 'model.vae_decoder.decoder.up_blocks.2.1.norm.bias', 'model.vae_decoder.decoder.up_blocks.2.2.conv1.weight', 'model.vae_decoder.decoder.up_blocks.2.2.conv1.bias', 'model.vae_decoder.decoder.up_blocks.2.2.conv2.weight', 'model.vae_decoder.decoder.up_blocks.2.2.norm.weight', 'model.vae_decoder.decoder.up_blocks.2.2.norm.bias', 'model.vae_decoder.decoder.up_blocks.2.3.conv1.weight', 'model.vae_decoder.decoder.up_blocks.2.3.conv1.bias', 'model.vae_decoder.decoder.up_blocks.2.3.conv2.weight', 'model.vae_decoder.decoder.up_blocks.2.3.norm.weight', 'model.vae_decoder.decoder.up_blocks.2.3.norm.bias', 'model.vae_decoder.decoder.up_blocks.3.0.conv.weight', 'model.vae_decoder.decoder.up_blocks.3.0.conv.bias', 'model.vae_decoder.decoder.up_blocks.3.1.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.3.1.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.3.1.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.3.1.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.3.1.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.3.1.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.3.1.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.3.1.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.3.1.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.3.1.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.3.1.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.3.1.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.3.1.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.3.1.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.3.1.conv_out.norm.bias', 'model.vae_decoder.decoder.up_blocks.3.2.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.3.2.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.3.2.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.3.2.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.3.2.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.3.2.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.3.2.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.3.2.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.3.2.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.3.2.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.3.2.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.3.2.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.3.2.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.3.2.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.3.2.conv_out.norm.bias', 'model.vae_decoder.decoder.up_blocks.3.3.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.3.3.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.3.3.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.3.3.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.3.3.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.3.3.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.3.3.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.3.3.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.3.3.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.3.3.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.3.3.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.3.3.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.3.3.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.3.3.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.3.3.conv_out.norm.bias', 'model.vae_decoder.decoder.up_blocks.4.0.conv.weight', 'model.vae_decoder.decoder.up_blocks.4.0.conv.bias', 'model.vae_decoder.decoder.up_blocks.4.1.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.4.1.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.4.1.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.4.1.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.4.1.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.4.1.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.4.1.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.4.1.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.4.1.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.4.1.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.4.1.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.4.1.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.4.1.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.4.1.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.4.1.conv_out.norm.bias', 'model.vae_decoder.decoder.up_blocks.4.2.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.4.2.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.4.2.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.4.2.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.4.2.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.4.2.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.4.2.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.4.2.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.4.2.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.4.2.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.4.2.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.4.2.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.4.2.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.4.2.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.4.2.conv_out.norm.bias', 'model.vae_decoder.decoder.up_blocks.4.3.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.4.3.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.4.3.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.4.3.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.4.3.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.4.3.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.4.3.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.4.3.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.4.3.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.4.3.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.4.3.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.4.3.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.4.3.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.4.3.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.4.3.conv_out.norm.bias', 'model.vae_decoder.decoder.up_blocks.5.0.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.5.0.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.5.0.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.5.0.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.5.0.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.5.0.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.5.0.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.5.0.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.5.0.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.5.0.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.5.0.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.5.0.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.5.0.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.5.0.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.5.0.conv_out.norm.bias', 'model.vae_decoder.decoder.up_blocks.5.1.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.5.1.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.5.1.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.5.1.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.5.1.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.5.1.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.5.1.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.5.1.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.5.1.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.5.1.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.5.1.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.5.1.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.5.1.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.5.1.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.5.1.conv_out.norm.bias', 'model.vae_decoder.decoder.up_blocks.5.2.attn.to_q.weight', 'model.vae_decoder.decoder.up_blocks.5.2.attn.to_k.weight', 'model.vae_decoder.decoder.up_blocks.5.2.attn.to_v.weight', 'model.vae_decoder.decoder.up_blocks.5.2.attn.to_qkv_multiscale.0.proj_in.weight', 'model.vae_decoder.decoder.up_blocks.5.2.attn.to_qkv_multiscale.0.proj_out.weight', 'model.vae_decoder.decoder.up_blocks.5.2.attn.to_out.weight', 'model.vae_decoder.decoder.up_blocks.5.2.attn.norm_out.weight', 'model.vae_decoder.decoder.up_blocks.5.2.attn.norm_out.bias', 'model.vae_decoder.decoder.up_blocks.5.2.conv_out.conv_inverted.weight', 'model.vae_decoder.decoder.up_blocks.5.2.conv_out.conv_inverted.bias', 'model.vae_decoder.decoder.up_blocks.5.2.conv_out.conv_depth.weight', 'model.vae_decoder.decoder.up_blocks.5.2.conv_out.conv_depth.bias', 'model.vae_decoder.decoder.up_blocks.5.2.conv_out.conv_point.weight', 'model.vae_decoder.decoder.up_blocks.5.2.conv_out.norm.weight', 'model.vae_decoder.decoder.up_blocks.5.2.conv_out.norm.bias', 'model.vae_decoder.decoder.norm_out.weight', 'model.vae_decoder.decoder.norm_out.bias', 'model.vae_decoder.decoder.conv_out.weight', 'model.vae_decoder.decoder.conv_out.bias', 'model.vae_decoder.down_blocks.0.mlp.0.weight', 'model.vae_decoder.down_blocks.0.mlp.0.bias', 'model.vae_decoder.down_blocks.0.mlp.1.weight', 'model.vae_decoder.down_blocks.0.mlp.1.bias', 'model.vae_decoder.down_blocks.0.mlp.3.weight', 'model.vae_decoder.down_blocks.0.mlp.3.bias', 'model.vae_decoder.down_blocks.1.mlp.0.weight', 'model.vae_decoder.down_blocks.1.mlp.0.bias', 'model.vae_decoder.down_blocks.1.mlp.1.weight', 'model.vae_decoder.down_blocks.1.mlp.1.bias', 'model.vae_decoder.down_blocks.1.mlp.3.weight', 'model.vae_decoder.down_blocks.1.mlp.3.bias', 'model.vae_decoder.down_blocks.2.mlp.0.weight', 'model.vae_decoder.down_blocks.2.mlp.0.bias', 'model.vae_decoder.down_blocks.2.mlp.1.weight', 'model.vae_decoder.down_blocks.2.mlp.1.bias', 'model.vae_decoder.down_blocks.2.mlp.3.weight', 'model.vae_decoder.down_blocks.2.mlp.3.bias', 'model.vae_decoder.down_mlp.0.weight', 'model.vae_decoder.down_mlp.0.bias', 'model.vae_decoder.down_mlp.1.weight', 'model.vae_decoder.down_mlp.1.bias', 'model.vae_decoder.down_mlp.3.weight', 'model.vae_decoder.down_mlp.3.bias',
## 'model.llm_connector
'model.llm_connector.layers.0.self_attn.q_proj.weight', 'model.llm_connector.layers.0.self_attn.q_proj.bias', 'model.llm_connector.layers.0.self_attn.k_proj.weight', 'model.llm_connector.layers.0.self_attn.k_proj.bias', 'model.llm_connector.layers.0.self_attn.v_proj.weight', 'model.llm_connector.layers.0.self_attn.v_proj.bias', 'model.llm_connector.layers.0.self_attn.o_proj.weight', 'model.llm_connector.layers.0.mlp.gate_proj.weight', 'model.llm_connector.layers.0.mlp.up_proj.weight', 'model.llm_connector.layers.0.mlp.down_proj.weight', 'model.llm_connector.layers.0.input_layernorm.weight', 'model.llm_connector.layers.0.post_attention_layernorm.weight', 'model.llm_connector.layers.1.self_attn.q_proj.weight', 'model.llm_connector.layers.1.self_attn.q_proj.bias', 'model.llm_connector.layers.1.self_attn.k_proj.weight', 'model.llm_connector.layers.1.self_attn.k_proj.bias', 'model.llm_connector.layers.1.self_attn.v_proj.weight', 'model.llm_connector.layers.1.self_attn.v_proj.bias', 'model.llm_connector.layers.1.self_attn.o_proj.weight', 'model.llm_connector.layers.1.mlp.gate_proj.weight', 'model.llm_connector.layers.1.mlp.up_proj.weight', 'model.llm_connector.layers.1.mlp.down_proj.weight', 'model.llm_connector.layers.1.input_layernorm.weight', 'model.llm_connector.layers.1.post_attention_layernorm.weight', 'model.llm_connector.layers.2.self_attn.q_proj.weight', 'model.llm_connector.layers.2.self_attn.q_proj.bias', 'model.llm_connector.layers.2.self_attn.k_proj.weight', 'model.llm_connector.layers.2.self_attn.k_proj.bias', 'model.llm_connector.layers.2.self_attn.v_proj.weight', 'model.llm_connector.layers.2.self_attn.v_proj.bias', 'model.llm_connector.layers.2.self_attn.o_proj.weight', 'model.llm_connector.layers.2.mlp.gate_proj.weight', 'model.llm_connector.layers.2.mlp.up_proj.weight', 'model.llm_connector.layers.2.mlp.down_proj.weight', 'model.llm_connector.layers.2.input_layernorm.weight', 'model.llm_connector.layers.2.post_attention_layernorm.weight', 'model.llm_connector.layers.3.self_attn.q_proj.weight', 'model.llm_connector.layers.3.self_attn.q_proj.bias', 'model.llm_connector.layers.3.self_attn.k_proj.weight', 'model.llm_connector.layers.3.self_attn.k_proj.bias', 'model.llm_connector.layers.3.self_attn.v_proj.weight', 'model.llm_connector.layers.3.self_attn.v_proj.bias', 'model.llm_connector.layers.3.self_attn.o_proj.weight', 'model.llm_connector.layers.3.mlp.gate_proj.weight', 'model.llm_connector.layers.3.mlp.up_proj.weight', 'model.llm_connector.layers.3.mlp.down_proj.weight', 'model.llm_connector.layers.3.input_layernorm.weight', 'model.llm_connector.layers.3.post_attention_layernorm.weight', 'model.llm_connector.layers.4.self_attn.q_proj.weight', 'model.llm_connector.layers.4.self_attn.q_proj.bias', 'model.llm_connector.layers.4.self_attn.k_proj.weight', 'model.llm_connector.layers.4.self_attn.k_proj.bias', 'model.llm_connector.layers.4.self_attn.v_proj.weight', 'model.llm_connector.layers.4.self_attn.v_proj.bias', 'model.llm_connector.layers.4.self_attn.o_proj.weight', 'model.llm_connector.layers.4.mlp.gate_proj.weight', 'model.llm_connector.layers.4.mlp.up_proj.weight', 'model.llm_connector.layers.4.mlp.down_proj.weight', 'model.llm_connector.layers.4.input_layernorm.weight', 'model.llm_connector.layers.4.post_attention_layernorm.weight', 'model.llm_connector.layers.5.self_attn.q_proj.weight', 'model.llm_connector.layers.5.self_attn.q_proj.bias', 'model.llm_connector.layers.5.self_attn.k_proj.weight', 'model.llm_connector.layers.5.self_attn.k_proj.bias', 'model.llm_connector.layers.5.self_attn.v_proj.weight', 'model.llm_connector.layers.5.self_attn.v_proj.bias', 'model.llm_connector.layers.5.self_attn.o_proj.weight', 'model.llm_connector.layers.5.mlp.gate_proj.weight', 'model.llm_connector.layers.5.mlp.up_proj.weight', 'model.llm_connector.layers.5.mlp.down_proj.weight', 'model.llm_connector.layers.5.input_layernorm.weight', 'model.llm_connector.layers.5.post_attention_layernorm.weight', 'model.llm_connector.norm.weight',
## 'model.projector
'model.projector.weight', 'model.projector.bias',
## 'lm_head
'lm_head.weight'])





# self.text_tokenizer.decode(input_ids[0])
"<|im_start|>user\nTask: Generate a First-Person View (FPV) image of CS2 map 'de_dust2' based on the Radar Map and Camera Pose.\nCoordinate System Definition:\n- Map Size: 1024x1024 pixels.\n- Yaw: 0 degrees is East, increases Clockwise.\n- Pitch: 0 degrees is looking straight Down (at feet), 180 degrees is looking straight Up (at sky).\n- Z-Height: Absolute vertical coordinate. Valid values are bounded by the map's global topology, ranging from the lowest point at -20.00 to the highest point at 56.00.\n\nCurrent Camera Pose: Position(x=486.0, y=628.0, z=13.000), Rotation(pitch=88.3, yaw=28.9)\n<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT></img><|im_end|>\n<|im_start|>assistant\n<img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><|endoftext|>"




# model_args
ModelArguments(model_name_or_path='UniLIP-1B', version='internvl', freeze_backbone=True, tune_mm_mlp_adapter=False, vision_tower=None, gen_vision_tower=None, mm_vision_select_layer=-1, pretrain_mm_mlp_adapter=None, pretrain_gen_mlp_adapter=None, vision_tower_pretrained=None, mm_projector_type='linear', gen_projector_type='linear', mm_use_im_start_end=False, mm_use_im_patch_token=False, mm_patch_merge_type='flat', mm_vision_select_feature='patch', n_query=256, n_und_query=0, gen_pooling='all', unilip_path='', unilip_factor=10.6, weighting_scheme='logit_normal', fix_dit=False, fix_connect=False, fix_vit=True, fix_llm=True, connect_layer=6, mllm_path='', mllm_hf_path='OpenGVLab/InternVL3-1B-hf', vae_path='', dit_path='', action_connect_layer=3)




# self.model.llm_connector
Qwen2Model(
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





# self.model.action_dit
Qwen2Model(
  (layers): ModuleList(
    (0-2): 3 x Qwen2DecoderLayer(
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





# Exp10 Model Before LoRA

## Unified_UniLIP_InternVLForCausalLM(
  ### (model): Unified_UniLIP_InternVLModel(
  #### (vision_tower): InternVisionModel(
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
  #### (multi_modal_projector): Sequential(
      (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
      (1): Linear(in_features=4096, out_features=896, bias=True)
      (2): GELU(approximate='none')
      (3): Linear(in_features=896, out_features=896, bias=True)
    )
  #### (language_model): Qwen2Model(
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
  #### (dit): SanaTransformer2DModel(
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
  #### (vae_decoder): DCAE_Decoder(
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
  #### (llm_connector): Qwen2Model(
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
  #### (projector): Linear(in_features=896, out_features=2304, bias=True)
  #### (action_dit_norm): Qwen2RMSNorm((896,), eps=1e-06)
  #### (action_dit): Qwen2Model(
      (layers): ModuleList(
        (0-2): 3 x Qwen2DecoderLayer(
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
          (input_layernorm): Qwen2RMSNormAdaRMS(
            (linear): Linear(in_features=896, out_features=2688, bias=True)
          )
          (post_attention_layernorm): Qwen2RMSNormAdaRMS(
            (linear): Linear(in_features=896, out_features=2688, bias=True)
          )
        )
      )
      (norm): Qwen2RMSNormAdaRMS(
        (linear): Linear(in_features=896, out_features=2688, bias=True)
      )
      (rotary_emb): Qwen2RotaryEmbedding()
    )
  #### (action_in_proj): Linear(in_features=5, out_features=896, bias=True)
  #### (time_mlp_in): Linear(in_features=896, out_features=896, bias=True)
  #### (time_mlp_out): Linear(in_features=896, out_features=896, bias=True)
  #### (action_out_proj): Linear(in_features=896, out_features=5, bias=True)
  #### (action_dit_projector): Sequential(
      (0): Linear(in_features=896, out_features=3584, bias=True)
      (1): GELU(approximate='none')
      (2): Linear(in_features=3584, out_features=1792, bias=True)
      (3): GELU(approximate='none')
      (4): Linear(in_features=1792, out_features=896, bias=True)
    )
  )
  ### (lm_head): Linear(in_features=896, out_features=151678, bias=False)
)
