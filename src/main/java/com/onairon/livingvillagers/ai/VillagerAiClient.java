package com.onairon.livingvillagers.ai;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.onairon.livingvillagers.OnaironsLivingVillagers;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

public class VillagerAiClient {
	private static final URI RESPOND_URI = URI.create("http://127.0.0.1:8000/villager/respond");
	private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(30L);

	private final HttpClient httpClient = HttpClient.newBuilder()
			.connectTimeout(Duration.ofSeconds(2L))
			.build();

	public CompletableFuture<VillagerAiResponse> requestResponse(SpeechSegmentRequest request) {
		var body = toJson(request).toString();
		var httpRequest = HttpRequest.newBuilder(RESPOND_URI)
				.version(HttpClient.Version.HTTP_1_1)
				.timeout(REQUEST_TIMEOUT)
				.header("Content-Type", "application/json")
				.POST(HttpRequest.BodyPublishers.ofString(body))
				.build();

		return httpClient.sendAsync(httpRequest, HttpResponse.BodyHandlers.ofString())
				.thenApply(response -> {
					if (response.statusCode() < 200 || response.statusCode() >= 300) {
						throw new IllegalStateException("AI server returned HTTP " + response.statusCode() + ": " + response.body());
					}

					return parseResponse(response.body());
				});
	}

	private JsonObject toJson(SpeechSegmentRequest request) {
		var root = new JsonObject();
		root.addProperty("event", "player_spoke");
		root.addProperty("transcript", "mock transcript");
		root.addProperty("audio_duration_ms", request.durationMs());
		root.addProperty("packet_count", request.packetCount());
		root.addProperty("opus_byte_count", request.opusByteCount());
		root.addProperty("whispering", request.whispering());
		root.addProperty("audio_sample_rate_hz", request.audioSampleRateHz());
		root.addProperty("audio_channels", request.audioChannels());
		root.addProperty("audio_pcm_s16le_base64", request.pcmAudioBase64());

		var player = new JsonObject();
		player.addProperty("uuid", request.playerUuid().toString());
		player.addProperty("name", request.playerName());
		player.addProperty("x", request.playerX());
		player.addProperty("y", request.playerY());
		player.addProperty("z", request.playerZ());
		root.add("player", player);

		var villager = new JsonObject();
		villager.addProperty("uuid", request.villagerUuid().toString());
		villager.addProperty("profession", request.villagerProfession());
		villager.addProperty("distance", request.villagerDistance());
		root.add("villager", villager);

		var scene = new com.google.gson.JsonArray();
		for (var fact : request.scene()) {
			scene.add(fact);
		}
		root.add("scene", scene);

		return root;
	}

	private VillagerAiResponse parseResponse(String body) {
		var root = JsonParser.parseString(body).getAsJsonObject();
		return new VillagerAiResponse(
				getString(root, "line", "..."),
				getString(root, "emotion", "neutral"),
				getString(root, "action", "none"),
				getString(root, "memory_update", "")
		);
	}

	private static String getString(JsonObject root, String name, String fallback) {
		if (!root.has(name) || root.get(name).isJsonNull()) {
			return fallback;
		}

		return root.get(name).getAsString();
	}

	public record SpeechSegmentRequest(
			UUID playerUuid,
			String playerName,
			double playerX,
			double playerY,
			double playerZ,
			UUID villagerUuid,
			String villagerProfession,
			double villagerDistance,
			List<String> scene,
			int packetCount,
			int opusByteCount,
			long durationMs,
			boolean whispering,
			int audioSampleRateHz,
			int audioChannels,
			String pcmAudioBase64
	) {
	}

	public record VillagerAiResponse(String line, String emotion, String action, String memoryUpdate) {
	}
}
