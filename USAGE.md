0. Create a VM

``` bash
terraform init
terraform plan
terraform apply
```

1. Edit inventory file with your server details got from terraform

``` bash
cd ansible
vim inventory.yml
```
2. Start the provision
``` bash
ansible-playbook -i inventory.ini playbook.yml
```

3. SSH to your server and activate environment
``` bash
ssh your-server
./activate_speecht5.sh
# OR
speecht5  # (alias)
# OR  
source speecht5_env/bin/activate
```
4. Verify everything works (should not be necessary)
``` bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```
5. Run your training
```
python train_speecht5.py --yes
```
