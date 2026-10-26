"""Optional real Transformers/PEFT CPU API roundtrip; no pretrained weights/GPU."""
import importlib.metadata
import tempfile
from pathlib import Path
import sys
import unittest

from _layout import PROJECT_ROOT
sys.path.insert(0,str(PROJECT_ROOT/'01_학습용'))
from train_lora import MODEL_REPO, MODEL_REVISION, target_pattern

try:
    READY = (importlib.metadata.version('transformers') == '4.57.1'
             and importlib.metadata.version('peft') == '0.17.1')
except importlib.metadata.PackageNotFoundError:
    READY = False


@unittest.skipUnless(READY, 'Requires the separate Transformers 4.57.1/PEFT 0.17.1 training environment.')
class TinyModelRoundtrip(unittest.TestCase):
    def test_trainer_adapter_and_full_model_roundtrip(self):
        import torch
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(42)
        config=Qwen3VLConfig(
            text_config={'vocab_size':128,'hidden_size':128,'intermediate_size':256,
                         'num_hidden_layers':1,'num_attention_heads':1,'num_key_value_heads':1,
                         'head_dim':128,'pad_token_id':0,'eos_token_id':1,
                         'rope_scaling':{'rope_type':'default','mrope_section':[24,20,20],'mrope_interleaved':True}},
            vision_config={'depth':1,'hidden_size':32,'intermediate_size':64,'num_heads':4,
                           'out_hidden_size':128,'deepstack_visual_indexes':[],'patch_size':2,'spatial_merge_size':2},
            image_token_id=120,video_token_id=121,vision_start_token_id=122,vision_end_token_id=123)
        base=Qwen3VLForConditionalGeneration(config)
        base.name_or_path=MODEL_REPO
        names=[name for name,_ in base.named_modules() if 'language_model.layers.' in name
               and name.endswith(('q_proj','k_proj','v_proj','o_proj'))]
        self.assertEqual(len(names),4)
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=target_pattern(names),
                                           bias='none',task_type='CAUSAL_LM',revision=MODEL_REVISION))
        tokens=torch.randint(2,100,(8,)).tolist()
        data=[{'input_ids':tokens,'labels':tokens,'attention_mask':[1]*8}]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=TrainingArguments(output_dir=str(root/'run'),max_steps=1,per_device_train_batch_size=1,
                                   use_cpu=True,report_to='none',save_strategy='no',logging_strategy='no',disable_tqdm=True)
            trainer=Trainer(model=model,args=args,train_dataset=data)
            trainer.train()
            model.eval()
            inputs=torch.tensor([tokens])
            with torch.no_grad():before=model(input_ids=inputs).logits
            model.save_pretrained(root/'adapter',safe_serialization=True)
            self.assertTrue((root/'adapter/adapter_model.safetensors').is_file())
            self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if 'visual' in n))
            merged=model.merge_and_unload(safe_merge=True)
            with torch.no_grad():after=merged(input_ids=inputs).logits
            torch.testing.assert_close(before,after,atol=1e-4,rtol=1e-4)
            merged.save_pretrained(root/'merged',safe_serialization=True)
            restored=Qwen3VLForConditionalGeneration.from_pretrained(root/'merged',local_files_only=True)
            restored.eval()
            with torch.no_grad():reloaded=restored(input_ids=inputs).logits
            torch.testing.assert_close(after,reloaded,atol=1e-4,rtol=1e-4)


if __name__=='__main__':unittest.main()
