# Controlla i dischi collegati
lsblk

# Il disco cache apparirà come /dev/sdb o simile
# Dovrai poi montarlo manualmente:
sudo mkfs.ext4 /dev/sdb
sudo mkdir /mnt/training-data
sudo mount /dev/sdb /mnt/training-data

# Per montaggio permanente, aggiungi a /etc/fstab:
echo '/dev/sdb /mnt/training-data ext4 defaults 0 2' | sudo tee -a /etc/fstab