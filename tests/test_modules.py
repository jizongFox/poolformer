from unittest import TestCase

import torch
from torch import nn

from models.metaformer import (
    Attention,
    AddPositionEmb,
    SpatialFc,
    metaformer_ppaa_s12_224,
    basic_blocks as meta_basic_blocks,
    MetaFormerBlock,
)
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
        self.input_image = torch.randn(10, 3, 224, 224)

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
            dim=3, pool_size=3, mlp_ratio=4.0, drop=0.0, drop_path=0.2
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

    def test_pos_embed(self):
        pos_embed = AddPositionEmb(dim=384, spatial_shape=(14, 14))
        patch_embed = PatchEmbed(
            patch_size=16, stride=16, embed_dim=384, norm_layer=LayerNormChannel
        )

        output = pos_embed(patch_embed(self.input_image))
        assert output.shape == (1, 384, 14, 14)

    def test_spatialfc(self):
        patch_embed = PatchEmbed(
            patch_size=16, stride=16, embed_dim=384, norm_layer=LayerNormChannel
        )
        spatial_fc = SpatialFc(dim=384, spatial_shape=(14, 14))
        output = spatial_fc(patch_embed(self.input_image))
        assert output.shape == (1, 384, 14, 14)


class TestMetaFormer(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.input_image = torch.randn(1, 3, 224, 224)

    def test_meta_former(self):
        model = metaformer_ppaa_s12_224(pretrained=True)
        print(model)

    def test_module_with_pooling_as_token_mixture(self):
        stage = meta_basic_blocks(
            dim=64, index=0, token_mixer=nn.Identity, layers=[2, 3, 4, 5]
        )
        print(stage)

    @torch.no_grad()
    def test_metablock_with_token_mixture(self):
        block = MetaFormerBlock(
            dim=3,
            token_mixer=lambda dim: nn.Conv2d(dim, dim, kernel_size=5, padding=2),
            use_layer_scale=False,
        )
        input = torch.randn(1, 3, 60, 60)
        output = block(input)
