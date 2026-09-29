# Organizer contract — frozen interpretation

This file freezes the participant-side implementation of the organizer's published answers.

## Geometry / scale
DICOM files may not contain PixelSpacing. The organizer confirmed one scanner and a fixed pixel size for the dataset:
- X: 0.60 mm/pixel
- Y: 1.05 mm/pixel

Centimetre-based FOV/ROI checks use the fixed acquisition scale. The axis expert converts pixel slope to physical angle; the ROI expert adds 3 cm Y / 2 cm X margin features. This is a coverage proxy rather than pixel-accurate ROI segmentation, and inference does not require PixelSpacing tags.

## violation_type
Exact allowed values:

Spine:
- `Некорректная укладка`
- `Не выравнена ось позвоночника`
- `Присутствуют посторонние предметы`

Proximal femur:
- `Некорректная укладка`
- `Некорректная область интереса`

Multiple violations are joined by `; `. No violations means an empty field.

Hip rotation is an internal positioning subcriterion. It is **not** emitted as its own `violation_type` because that value is not present in the published closed list.

## quality_prob
The participant output includes:

`quality_prob` — probability of a quality violation, in `[0,1]`.

## anatomical_region
Only these values are emitted:

- `Поясничный отдел позвоночника`
- `Проксимальный отдел бедра`

Side is ignored.

## Hidden closed test
The organizer states that the final closed test set is not transmitted to participants. The participant-visible `Для теста.zip` is used only for format/debug validation. No hidden-test score is claimed in the release package.
