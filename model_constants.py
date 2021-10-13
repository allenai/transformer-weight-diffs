T5_ENCODER_STORE = {
    "q": 'encoder.block.{}.layer.0.SelfAttention.q.weight',
    "k": 'encoder.block.{}.layer.0.SelfAttention.k.weight',
    "v": 'encoder.block.{}.layer.0.SelfAttention.v.weight',
    "o": 'encoder.block.{}.layer.0.SelfAttention.o.weight',
    "wi": 'encoder.block.{}.layer.1.DenseReluDense.wi.weight',
    "wo": 'encoder.block.{}.layer.1.DenseReluDense.wo.weight',
}

T5_DECODER_STORE = {
    "q": 'decoder.block.{}.layer.0.SelfAttention.q.weight',
    "k": 'decoder.block.{}.layer.0.SelfAttention.k.weight',
    "v": 'decoder.block.{}.layer.0.SelfAttention.v.weight',
    "o": 'decoder.block.{}.layer.0.SelfAttention.o.weight',
    "xq": 'decoder.block.{}.layer.1.EncDecAttention.q.weight',
    "xk": 'decoder.block.{}.layer.1.EncDecAttention.k.weight',
    "xv": 'decoder.block.{}.layer.1.EncDecAttention.v.weight',
    "xo": 'decoder.block.{}.layer.1.EncDecAttention.o.weight',
    "wi": 'decoder.block.{}.layer.2.DenseReluDense.wi.weight',
    "wo": 'decoder.block.{}.layer.2.DenseReluDense.wo.weight',
}

MODELS_TO_STORE = {
    "t5-large": {
        "encoder": T5_ENCODER_STORE,
        "decoder": T5_DECODER_STORE
    }
}

MODELS_TO_LAYERS = {
    "t5-large": 24
}