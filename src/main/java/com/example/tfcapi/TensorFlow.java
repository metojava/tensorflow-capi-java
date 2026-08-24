package com.example.tfcapi;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

/**
 * Loads the native bridge (tf_capi_jni.dll) and the TensorFlow C library
 * (tensorflow.dll) that it depends on.
 *
 * <p>The library is fully self-contained: both DLLs are packed inside this
 * JAR under {@code /native/windows-x86_64/} and are extracted on first use to
 * {@code %USERPROFILE%\.tfcapi\cache\windows-x86_64\}. No PATH or environment
 * setup is required by consumers.</p>
 *
 * <p>Resolution order:</p>
 * <ol>
 *   <li>{@code TF_CAPI_NATIVE_DIR} environment variable pointing at a folder
 *       containing both DLLs (overrides everything, useful for development);</li>
 *   <li>natives bundled in the classpath (the normal case);</li>
 *   <li>development fallbacks relative to the project root
 *       ({@code native/dist/lib/tensorflow.dll} and
 *       {@code target/native/tf_capi_jni.dll}).</li>
 * </ol>
 *
 * <p>If you upgrade the vendored TensorFlow version, delete the cache folder
 * once so the new DLL is extracted.</p>
 */
public final class TensorFlow {

    private static volatile boolean loaded;

    private TensorFlow() {}

    /** TensorFlow C API version string, e.g. "2.18.1". Loads natives on first use. */
    public static String version() {
        load();
        return nativeVersion();
    }

    private static native String nativeVersion();

    public static synchronized void load() {
        if (loaded) {
            return;
        }
        try {
            Path dir = nativeLibraryDir();
            // Load tensorflow.dll first so the JNI bridge's dependency resolves
            // even when the extraction/cache folder is not on PATH.
            System.load(dir.resolve("tensorflow.dll").toAbsolutePath().toString());
            System.load(dir.resolve("tf_capi_jni.dll").toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            UnsatisfiedLinkError err = new UnsatisfiedLinkError(
                    "Could not load TensorFlow natives: " + e.getMessage());
            err.initCause(e);
            throw err;
        }
    }

    private static Path nativeLibraryDir() throws IOException {
        String override = System.getenv("TF_CAPI_NATIVE_DIR");
        if (override != null && !override.isBlank()) {
            Path dir = Paths.get(override);
            requireDlls(dir, "TF_CAPI_NATIVE_DIR");
            return dir;
        }

        Path cache = Paths.get(System.getProperty("user.home"),
                ".tfcapi", "cache", "windows-x86_64");
        Files.createDirectories(cache);

        boolean bundled = extractFromclasspath("tensorflow.dll",
                cache.resolve("tensorflow.dll"), false);
        // The JNI bridge is small and changes on every rebuild: always refresh it.
        boolean jniBundled = extractFromclasspath("tf_capi_jni.dll",
                cache.resolve("tf_capi_jni.dll"), true);

        if (!bundled || !jniBundled) {
            // Development mode: run from the project without a packaged jar.
            Path projectRoot = Paths.get("").toAbsolutePath();
            while (projectRoot != null && !Files.exists(projectRoot.resolve("native/dist/lib/tensorflow.dll"))
                    && !Files.exists(projectRoot.resolve("target/native/tf_capi_jni.dll"))) {
                projectRoot = projectRoot.getParent();
            }
            if (projectRoot == null) {
                throw new IOException(
                        "no natives found in classpath, TF_CAPI_NATIVE_DIR or project layout");
            }
            if (!bundled) {
                Files.copy(projectRoot.resolve("native/dist/lib/tensorflow.dll"),
                        cache.resolve("tensorflow.dll"), StandardCopyOption.REPLACE_EXISTING);
            }
            if (!jniBundled) {
                Files.copy(projectRoot.resolve("target/native/tf_capi_jni.dll"),
                        cache.resolve("tf_capi_jni.dll"), StandardCopyOption.REPLACE_EXISTING);
            }
        }
        return cache;
    }

    /**
     * Copies a bundled native out of the classpath to {@code target}.
     * Returns {@code true} when the resource exists (and therefore the target
     * is now populated), {@code false} when there is no such resource.
     */
    private static boolean extractFromclasspath(String name, Path target,
                                                boolean alwaysOverwrite) throws IOException {
        try (InputStream in = TensorFlow.class.getResourceAsStream(
                "/native/windows-x86_64/" + name)) {
            if (in == null) {
                return false;
            }
            if (!alwaysOverwrite && Files.exists(target) && Files.size(target) > 0) {
                return true;
            }
            Path tmp = target.resolveSibling(target.getFileName() + ".tmp");
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            Files.move(tmp, target,
                    StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            return true;
        }
    }

    private static void requireDlls(Path dir, String what) throws IOException {
        for (String dll : new String[]{"tensorflow.dll", "tf_capi_jni.dll"}) {
            if (!Files.isRegularFile(dir.resolve(dll))) {
                throw new IOException(what + " (" + dir + ") is missing " + dll);
            }
        }
    }
}
