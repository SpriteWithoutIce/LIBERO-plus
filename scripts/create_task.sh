# python scripts/generate_libero_object_swap_tasks.py \
#   --suite libero_object \
#   --task pick_up_the_alphabet_soup_and_place_it_in_the_basket \
#   --suffix swap_1 \
#   --seed 0 \
#   --manifest /home/jwhe/linyihan/LIBERO-plus/manifest/swap_manifest_swap_1.json \
#   --register-benchmark \
#   --classification-category "Objects Layout" \
#   --classification-difficulty 0 \
#   --update-task-num

python scripts/generate_pruned_init_from_bddl.py \
  --suite libero_object \
  --task pick_up_the_alphabet_soup_and_place_it_in_the_basket_swap_1 \
  --num-states 50 \
  --base-seed 0