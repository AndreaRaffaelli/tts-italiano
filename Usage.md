# 1. Edit inventory file with your server details
vim inventory.ini

# 2. Run the playbook
ansible-playbook -i inventory.ini cuda_setup.yml

# 3. SSH to your server and activate environment
ssh your-server
./activate_speecht5.sh
# OR
speecht5  # (alias)
# OR  
source speecht5_env/bin/activate

# 4. Verify everything works
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. Run your training
python train_speecht5.py --yes