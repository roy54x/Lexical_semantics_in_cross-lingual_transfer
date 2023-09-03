def get_cfg():
    cfg = {}
    cfg.update(use_tokenizer=True)
    cfg.update(tokenizer_name="tokenizer_en_el_cc100_clean")

    cfg.update(train_data_path="data/TED_data/TED_en-el_gt/train_set")
    cfg.update(test_data_path="data/TED_data/TED_en-el_gt/test_set")
    cfg.update(col_to_use="el_semantic_transformation_0entropy")
    cfg.update(train_data_size=50000)
    cfg.update(validation_size=0.1)
    cfg.update(test_data_size=100000)

    cfg.update(epochs_to_decide=5)
    cfg.update(batch_size=16)
    cfg.update(learning_rate=3e-5)
    cfg.update(num_hidden_layers=6)
    cfg.update(num_attention_heads=6)
    cfg.update(parallel_training=False)

    cfg.update(model_name="sentence_transformer_TED_en_el-semantic-0entropy_" + str(cfg["train_data_size"]) + "_" + str(cfg["num_hidden_layers"])
                          + "_layers" + "_" + str(cfg["num_attention_heads"]) + "_attention-heads")

    return cfg
