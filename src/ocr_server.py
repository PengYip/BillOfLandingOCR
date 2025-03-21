from pathlib import Path
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-StructureV3",device="gpu")

input_file = "./doc/pdf/bill/58730661-5 BL.pdf"
output_path = Path("./output")

output = pipeline.predict(
    input=input_file,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)