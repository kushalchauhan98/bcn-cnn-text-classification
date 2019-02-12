{
  "dataset_reader":{
    "type": "smile",
    "lazy": true,
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },
  "validation_dataset_reader":{
    "type": "quora",
    "lazy": true,
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "elmo": {
        "type": "elmo_characters"
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
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
            "type": "embedding",
            "embedding_dim": 50,
            "trainable": false
        }
      }
    },
    "embedding_dropout": 0.5,
    "pre_encode_feedforward": {
        "input_dim": 306,
        "num_layers": 1,
        "hidden_dims": [100],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 600,
      "hidden_size": 100,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "elmo": {
      "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
      "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
      "do_layer_norm": false,
      "dropout": 0.0,
      "num_output_representations": 1
    },
    "use_input_elmo": true,
    "use_integrator_output_elmo": false,
    "output_layer": {
        "input_dim": 800,
        "num_layers": 3,
        "output_dims": [400, 200, 7],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 50
  },
  "trainer": {
    "num_epochs": 20,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}
