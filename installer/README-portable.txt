GraphicsHost - portable package

Unzip anywhere (a folder you can write to is best) and start GraphicsHost.exe. Nothing is installed and nothing is written
outside the data folder (%USERPROFILE%\.graphicshost, or the folder in the GRAPHICSHOST_HOME variable).

  GraphicsHost.exe                  the window (the first start shows a short wizard)
  GraphicsHost.exe serve            only the HTTP server, no window (browser UI at http://127.0.0.1:1234/)
  GraphicsHost.exe service install  run the server at Windows start-up (administrator prompt needed)
  GraphicsHost.exe --help           the command line

You need a graphics card with a Vulkan driver (AMD) for fast images; without one the CPU runtime is used. Keep the folders
runtimes\, data\ and web\ next to the exe. See NOTICE.txt for the components inside.
