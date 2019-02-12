{
  "dataset_reader":{
    "type": "smile",
    "lazy":true,
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      }
    }
  },
  "validation_dataset_reader":{
    "type": "smile",
    "lazy": true,
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      }
    }
  },
  "train_data_path": "../train.csv",
  "validation_data_path": "../test.csv",
  "test_data_path": "../test.csv",
  "model": {
    "type": "bcn",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "embedding_dropout": 0.25,
    "pre_encode_feedforward": {
        "input_dim": 300,
        "num_layers": 1,
        "hidden_dims": [150],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 150,
      "hidden_size": 150,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 900,
      "hidden_size": 150,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "output_layer": {
        "input_dim": 1200,
        "num_layers": 3,
        "output_dims": [600, 300, 7],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 100
  },
  "trainer": {
    "num_epochs": 15,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.0005
    },
  "learning_rate_scheduler": {
    "type": "reduce_on_plateau",
    "factor": 0.5,
    "mode": "max",
    "patience": 0
    }
  }
}
