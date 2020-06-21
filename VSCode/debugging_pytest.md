Hi All! 

For many of these assignments, using a visual debugger will make your life much easier. Here is a tutorial for debugging your PyTest tests in VSCode. 

1. Install the Python Extension. Simply search Python in the extensions menu. You should see something like this from Microsoft:
![Screenshot%20from%202020-06-21%2016-56-03.png](https://campuspro-uploads.s3.us-west-2.amazonaws.com/63aa7cea-5e9c-4b62-96b7-8bbf3bc31b76/41546035-a0cc-47d4-96cd-a97d179a1b84/Screenshot%20from%202020-06-21%2016-56-03.png)
After installing you should restart VSCode.
2. Configure your Python interpreter. After successfully installing the Python extension, you should see this icon in the bottom left hand corner of the VSCode window.
![Screenshot%20from%202020-06-21%2017-09-12.png](https://campuspro-uploads.s3.us-west-2.amazonaws.com/63aa7cea-5e9c-4b62-96b7-8bbf3bc31b76/a65ba072-252c-46a4-ac08-b7e93229591a/Screenshot%20from%202020-06-21%2017-09-12.png)
Press on this icon, then select your conda environment for this homework from the dropdown. If you do not see your conda environment, you will have to enter an interpreter path. This is usually `<conda_directory>/envs/<env_name>/bin/python`.  `conda_directory` is usually `anaconda3`, `anaconda` or `.conda` in your home directory. If you installed Miniconda, it may be `miniconda3` or `miniconda`. Windows users may have the first letter of the directory capitalized. See this article for more information about finding your conda path: [https://docs.anaconda.com/anaconda/user-guide/tasks/integration/python-path/](https://docs.anaconda.com/anaconda/user-guide/tasks/integration/python-path/)

3. Create the file `.vscode/launch.json` in the root directory of your homework. Make directories as necessary.
4. Copy and paste the following into `.vscode/launch.json`
 ```
{
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "PyTest",
            "type": "python",
            "module": "pytest",
            "args": ["${workspaceFolder}/tests/test_data.py::test_load_data"],
            "request": "launch",
            "console": "integratedTerminal"
        }
    ]
}
```
   This code will debug the test `test_load_data` in the test file `tests/test_data.py` in the VSCode terminal. To change the function being debugged, like for example you want to debug `test_train_test_split`, you would change the `args` argument such that `test_train_test_split` comes after the `::`:
```
{
 "args": ["${workspaceFolder}/tests/test_data.py::test_train_test_split"],
}
```
If you want to debug all tests in `tests/test_data.py`, you would remove the `::` and everything after it:
```
{
 "args": ["${workspaceFolder}/tests/test_data.py"],
}
```
If you would want to debug all tests, remove the `args` argument entirely.

5. Debug your tests. First you will want to set breakpoints by pressing to the left of the line numbers:
![Screenshot%20from%202020-06-21%2017-33-02.png](https://campuspro-uploads.s3.us-west-2.amazonaws.com/63aa7cea-5e9c-4b62-96b7-8bbf3bc31b76/030ed8c9-9ad1-4bb8-971d-cbaed86ab2fe/Screenshot%20from%202020-06-21%2017-33-02.png)
Then press this icon in the debug menu:
![Screenshot%20from%202020-06-21%2017-33-38.png](https://campuspro-uploads.s3.us-west-2.amazonaws.com/63aa7cea-5e9c-4b62-96b7-8bbf3bc31b76/fc938fc1-581b-4031-b6aa-87df58b10834/Screenshot%20from%202020-06-21%2017-33-38.png)
You should now be debugging! For more general information on how to use the debugger, see this article from Microsoft: [https://code.visualstudio.com/docs/editor/debugging](https://code.visualstudio.com/docs/editor/debugging)

Happy Debugging!

