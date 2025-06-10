## Note on Docker Image Setup

While building the Docker image, some required Python libraries failed to install correctly during the image creation process.

As a workaround, I used a minimal `Dockerfile` and `requirements.txt`, and then manually installed the missing libraries at runtime using the `runai` command:

```bash
runai submit --name gpt-v4-small \
  --run-as-user \
  --image registry.rcp.epfl.ch/ee-559-shayakhm/my-toolbox:v0.1 \
  --gpu 1 \
  --pvc home:/home/shayakhm \
  -e HOME=/home/shayakhm \
  --command -- bash -c "pip install transformers evaluate datasets tokenizers accelerate lime sentence-transformers rouge_score nltk bert_score && python3 /home/shayakhm/gpt_latent_py_v4_small/together.py"
```
