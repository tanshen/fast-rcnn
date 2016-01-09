for line in $(cat test.txt)
do
    cp ${line}.jpg ../train/${line}.jpg
done
