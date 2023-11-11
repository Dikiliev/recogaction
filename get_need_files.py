import os

def delete_files_except_first_30(folder_path):
    # Проверяем, существует ли указанный путь
    if not os.path.exists(folder_path):
        print("Указанный путь не существует.")
        return

    # Перебираем все подпапки в указанной папке
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)

        # Проверяем, является ли элемент подпапкой
        if os.path.isdir(subdir_path):
            files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            files.sort()  # Сортируем файлы по имени

            # Удаляем все файлы, кроме первых 30
            for file in files[20:]:
                os.remove(os.path.join(subdir_path, file))
                print(f"Удален файл: {file}")

# Использование функции
path = "C:\\Users\\besla\\Desktop\\videos"
delete_files_except_first_30(path)
