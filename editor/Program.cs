//
// Program.cs by Xein
// 12 May 2026
//

using System;
using Avalonia;
using editor.Engine;

namespace editor;

internal sealed class Program {
	[STAThread]
	public static void Main(string[] args) {
		BuildAvaloniaApp()
			.StartWithClassicDesktopLifetime(args);
	}

	public static AppBuilder BuildAvaloniaApp() {
		var builder = AppBuilder.Configure<App>()
			.UsePlatformDetect()
#if DEBUG
			.WithDeveloperTools()
#endif
			.WithInterFont();

		// RenderDoc hooks the first graphics API it sees process-wide; if it's already attached (editor
		// launched through RenderDoc), Avalonia's own hardware-accelerated renderer creating a second,
		// unrelated context is what crashes on startup. Software rendering sidesteps that entirely.
		// When RenderDoc isn't attached, use the normal hardware-accelerated path
		if (RenderDocDetector.IsAttached) {
			builder = builder
				.With(new Win32PlatformOptions {
					RenderingMode = [ Win32RenderingMode.Software ]
				})
				.With(new X11PlatformOptions {
					RenderingMode = [ X11RenderingMode.Software ]
				});
		}

		return builder.LogToTrace();
	}
}
