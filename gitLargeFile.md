git lfs install

git lfs track "*.csv"

git add .gitattributes

git add fileName.csv

git commit -m "Add design file"

git lfs migrate import --include="*.csv"                                         


git push --force