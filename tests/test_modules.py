from unittest import TestCase

import torch

from models.metaformer import Attention
from models.poolformer import (
    PatchEmbed,
    LayerNormChannel,
    Pooling,
    Mlp,
    PoolFormerBlock,
    basic_blocks,
    poolformer_s12,
)


class TestModules(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.input_image = torch.randn(1, 3, 224, 224)

    def test_patchembed(self):
        patch_size = 16
        emb_dim = 768

        norm_layer = LayerNormChannel

        embed = PatchEmbed(
            patch_size=patch_size,
            stride=patch_size,
            embed_dim=emb_dim,
            norm_layer=norm_layer,
        )
        output = embed(self.input_image)
        assert output.shape == (1, emb_dim, 224 / patch_size, 224 / patch_size)

    def test_pooling(self):
        pool_module = Pooling(pool_size=3)
        output = pool_module(self.input_image) + self.input_image
        assert output.shape == self.input_image.shape

    def test_mlp(self):
        out_feature = 200
        mlp_module = Mlp(
            in_features=3,
            hidden_features=300,
            out_features=out_feature,
        )
        output = mlp_module(self.input_image)
        assert output.shape == (1, 200, 224, 224)

    def test_poolformer_block(self):
        block = PoolFormerBlock(
            dim=3, pool_size=3, mlp_ratio=4.0, drop=0.0, drop_path=0.0
        )
        print(block)
        output = block(self.input_image)

    def test_blocks(self):
        embd = PatchEmbed(
            patch_size=16, stride=16, embed_dim=16, norm_layer=LayerNormChannel
        )
        blocks = basic_blocks(dim=16, index=3, layers=[2, 3, 4, 5])
        print(blocks(embd(self.input_image)).shape)

    def test_poolformer_s12(self):
        model = poolformer_s12()
        import warnings

        warnings.warn(
            "there is no class token and no positional embedding in the model"
        )
        output = model(self.input_image)
        print(output.shape)

    def test_MHA(self):
        model = Attention(dim=128, head_dim=16)
        embd = PatchEmbed(
            patch_size=16, stride=16, embed_dim=128, norm_layer=LayerNormChannel
        )
        output = model(embd(self.input_image))
        print(output.shape)
