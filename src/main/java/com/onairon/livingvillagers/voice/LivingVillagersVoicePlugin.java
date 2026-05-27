package com.onairon.livingvillagers.voice;

import com.onairon.livingvillagers.OnaironsLivingVillagers;
import com.onairon.livingvillagers.ai.VillagerAiClient;
import de.maxhenkel.voicechat.api.VoicechatPlugin;
import de.maxhenkel.voicechat.api.VoicechatServerApi;
import de.maxhenkel.voicechat.api.events.EventRegistration;
import de.maxhenkel.voicechat.api.events.MicrophonePacketEvent;
import de.maxhenkel.voicechat.api.events.VoicechatServerStartedEvent;
import de.maxhenkel.voicechat.api.opus.OpusDecoder;
import de.maxhenkel.voicechat.api.opus.OpusEncoderMode;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.entity.npc.villager.Villager;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.entity.EntityTypeTest;
import net.minecraft.world.phys.AABB;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class LivingVillagersVoicePlugin implements VoicechatPlugin {
	private static final long SPEECH_GAP_TIMEOUT_MS = 700L;
	private static final long MIN_SPEECH_DURATION_MS = 900L;
	private static final int MIN_SPEECH_OPUS_BYTES = 2500;
	private static final double VILLAGER_HEARING_RADIUS_BLOCKS = 8.0D;
	private static final int CONTEXT_SCAN_RADIUS_BLOCKS = 4;
	private static final int VOICECHAT_SAMPLE_RATE_HZ = 48_000;
	private static final int VOICECHAT_CHANNELS = 1;
	private static final float VILLAGER_VOICE_DISTANCE_BLOCKS = 16.0F;
	private final Map<UUID, SpeechSegmentBuffer> speechBuffers = new ConcurrentHashMap<>();
	private final VillagerAiClient aiClient = new VillagerAiClient();
	private volatile VoicechatServerApi voicechatApi;

	@Override
	public String getPluginId() {
		return OnaironsLivingVillagers.MOD_ID;
	}

	@Override
	public void registerEvents(EventRegistration registration) {
		registration.registerEvent(VoicechatServerStartedEvent.class, this::onVoicechatServerStarted);
		registration.registerEvent(MicrophonePacketEvent.class, this::onMicrophonePacket);
	}

	private void onVoicechatServerStarted(VoicechatServerStartedEvent event) {
		voicechatApi = event.getVoicechat();
		OnaironsLivingVillagers.LOGGER.info("Living Villagers voice plugin connected to Simple Voice Chat");
	}

	private void onMicrophonePacket(MicrophonePacketEvent event) {
		var sender = event.getSenderConnection();
		var player = sender == null ? null : sender.getPlayer();
		var packet = event.getPacket();

		if (player == null || packet == null) {
			return;
		}

		var playerUuid = player.getUuid();
		var opusData = packet.getOpusEncodedData();
		var buffer = speechBuffers.computeIfAbsent(playerUuid, uuid -> new SpeechSegmentBuffer());
		long now = System.currentTimeMillis();

		if (opusData.length == 0) {
			finishSpeechSegment(player, buffer, "empty packet");
			return;
		}

		if (buffer.hasAudio() && now - buffer.lastPacketAtMs() > SPEECH_GAP_TIMEOUT_MS) {
			finishSpeechSegment(player, buffer, "silence gap");
		}

		buffer.addPacket(opusData, packet.isWhispering(), now, voicechatApi);
	}

	private void finishSpeechSegment(de.maxhenkel.voicechat.api.ServerPlayer voicePlayer, SpeechSegmentBuffer buffer, String reason) {
		var segment = buffer.finish();

		if (segment.packetCount() <= 0) {
			return;
		}

		var playerUuid = voicePlayer.getUuid();
		OnaironsLivingVillagers.LOGGER.info(
				"Speech segment from {} complete: {} packets, {} opus bytes, {} ms, whispering={}, reason={}",
				playerUuid,
				segment.packetCount(),
				segment.opusByteCount(),
				segment.durationMs(),
				segment.whispering(),
				reason
		);

		if (segment.durationMs() < MIN_SPEECH_DURATION_MS || segment.opusByteCount() < MIN_SPEECH_OPUS_BYTES) {
			OnaironsLivingVillagers.LOGGER.info(
					"Ignoring short speech segment from {}: {} ms, {} opus bytes",
					playerUuid,
					segment.durationMs(),
					segment.opusByteCount()
			);
			return;
		}

		if (!(voicePlayer.getPlayer() instanceof ServerPlayer minecraftPlayer)) {
			OnaironsLivingVillagers.LOGGER.warn("Could not map voice player {} to Minecraft ServerPlayer", playerUuid);
			return;
		}

		var server = minecraftPlayer.level().getServer();
		if (server == null) {
			OnaironsLivingVillagers.LOGGER.warn("Could not access Minecraft server for player {}", playerUuid);
			return;
		}

		server.execute(() -> handleSpeechSegmentOnServerThread(minecraftPlayer, segment));
	}

	private void handleSpeechSegmentOnServerThread(ServerPlayer minecraftPlayer, SpeechSegment segment) {
		var nearestVillager = findNearestVillager(minecraftPlayer);

		if (nearestVillager == null) {
			OnaironsLivingVillagers.LOGGER.info(
					"Ignoring speech from {} because no villager is within {} blocks",
					minecraftPlayer.getGameProfile().name(),
					VILLAGER_HEARING_RADIUS_BLOCKS
			);
			return;
		}

		var distance = Math.sqrt(nearestVillager.distanceToSqr(minecraftPlayer));
		var profession = getVillagerProfession(nearestVillager);
		var dimension = minecraftPlayer.level().dimension().identifier().toString();
		var dayTime = minecraftPlayer.level().getDayTime() % 24000L;
		var scene = new ArrayList<String>();
		scene.add("Player spoke through Simple Voice Chat.");
		scene.add("Speech audio is attached for optional speech-to-text on the AI server.");
		scene.add(String.format("Nearest villager is %.1f blocks away.", distance));
		scene.add("Nearest villager profession is " + profession + ".");
		scene.add("Current dimension is " + dimension + ".");
		scene.add("World time is " + dayTime + " ticks (" + describeTimeOfDay(dayTime) + ").");
		scene.addAll(scanVillagerSurroundings(nearestVillager));

		OnaironsLivingVillagers.LOGGER.info("Sending villager scene context: {}", scene);

		aiClient.requestResponse(new VillagerAiClient.SpeechSegmentRequest(
						minecraftPlayer.getUUID(),
						minecraftPlayer.getGameProfile().name(),
						minecraftPlayer.getX(),
						minecraftPlayer.getY(),
						minecraftPlayer.getZ(),
						nearestVillager.getUUID(),
						profession,
						distance,
						scene,
						segment.packetCount(),
						segment.opusByteCount(),
						segment.durationMs(),
						segment.whispering(),
						VOICECHAT_SAMPLE_RATE_HZ,
						VOICECHAT_CHANNELS,
						segment.pcmAudioBase64()
				))
				.thenAccept(response -> {
					OnaironsLivingVillagers.LOGGER.info(
							"Villager AI response: line='{}', emotion={}, action={}",
							response.line(),
							response.emotion(),
							response.action()
					);

					var server = minecraftPlayer.level().getServer();
					if (server != null) {
						server.execute(() -> {
							if (!nearestVillager.isAlive()) {
								OnaironsLivingVillagers.LOGGER.info(
										"Skipping villager response because target villager {} is no longer alive",
										nearestVillager.getUUID()
								);
								return;
							}

							sendPlayerMessage(minecraftPlayer, "Villager: " + response.line());
							playVillagerVoice(nearestVillager, response.line());
						});
					}
				})
				.exceptionally(error -> {
					OnaironsLivingVillagers.LOGGER.warn("Could not get villager AI response", error);
					return null;
				});
	}

	private Villager findNearestVillager(ServerPlayer player) {
		var searchBox = new AABB(player.blockPosition()).inflate(VILLAGER_HEARING_RADIUS_BLOCKS);
		return player.level()
				.getEntities(EntityTypeTest.forClass(Villager.class), searchBox, Villager::isAlive)
				.stream()
				.min(Comparator.comparingDouble(villager -> villager.distanceToSqr(player)))
				.orElse(null);
	}

	private String getVillagerProfession(Villager villager) {
		return villager.getVillagerData()
				.profession()
				.unwrapKey()
				.map(key -> key.identifier().getPath())
				.map(path -> path.equals("none") ? "unemployed" : path.replace('_', ' '))
				.orElse("unknown");
	}

	private String describeTimeOfDay(long dayTime) {
		if (dayTime >= 23000L || dayTime < 1000L) {
			return "sunrise";
		}
		if (dayTime < 12000L) {
			return "day";
		}
		if (dayTime < 13000L) {
			return "sunset";
		}
		return "night";
	}

	private ArrayList<String> scanVillagerSurroundings(Villager villager) {
		var scene = new ArrayList<String>();
		var level = villager.level();
		var origin = villager.blockPosition();
		int beds = 0;
		int doors = 0;
		int crops = 0;
		int chests = 0;
		int jobSites = 0;

		for (int dx = -CONTEXT_SCAN_RADIUS_BLOCKS; dx <= CONTEXT_SCAN_RADIUS_BLOCKS; dx++) {
			for (int dy = -2; dy <= 2; dy++) {
				for (int dz = -CONTEXT_SCAN_RADIUS_BLOCKS; dz <= CONTEXT_SCAN_RADIUS_BLOCKS; dz++) {
					var blockState = level.getBlockState(origin.offset(dx, dy, dz));
					var block = blockState.getBlock();

					if (blockState.is(Blocks.WHITE_BED) || blockState.is(Blocks.ORANGE_BED) || blockState.is(Blocks.MAGENTA_BED)
							|| blockState.is(Blocks.LIGHT_BLUE_BED) || blockState.is(Blocks.YELLOW_BED) || blockState.is(Blocks.LIME_BED)
							|| blockState.is(Blocks.PINK_BED) || blockState.is(Blocks.GRAY_BED) || blockState.is(Blocks.LIGHT_GRAY_BED)
							|| blockState.is(Blocks.CYAN_BED) || blockState.is(Blocks.PURPLE_BED) || blockState.is(Blocks.BLUE_BED)
							|| blockState.is(Blocks.BROWN_BED) || blockState.is(Blocks.GREEN_BED) || blockState.is(Blocks.RED_BED)
							|| blockState.is(Blocks.BLACK_BED)) {
						beds++;
					}

					if (blockState.is(Blocks.OAK_DOOR) || blockState.is(Blocks.SPRUCE_DOOR) || blockState.is(Blocks.BIRCH_DOOR)
							|| blockState.is(Blocks.JUNGLE_DOOR) || blockState.is(Blocks.ACACIA_DOOR) || blockState.is(Blocks.DARK_OAK_DOOR)
							|| blockState.is(Blocks.MANGROVE_DOOR) || blockState.is(Blocks.CHERRY_DOOR) || blockState.is(Blocks.BAMBOO_DOOR)
							|| blockState.is(Blocks.CRIMSON_DOOR) || blockState.is(Blocks.WARPED_DOOR)) {
						doors++;
					}

					if (blockState.is(Blocks.WHEAT) || blockState.is(Blocks.CARROTS) || blockState.is(Blocks.POTATOES)
							|| blockState.is(Blocks.BEETROOTS)) {
						crops++;
					}

					if (blockState.is(Blocks.CHEST) || blockState.is(Blocks.TRAPPED_CHEST) || blockState.is(Blocks.BARREL)) {
						chests++;
					}

					if (blockState.is(Blocks.COMPOSTER) || blockState.is(Blocks.LECTERN) || blockState.is(Blocks.BLAST_FURNACE)
							|| blockState.is(Blocks.SMOKER) || blockState.is(Blocks.BREWING_STAND) || blockState.is(Blocks.CARTOGRAPHY_TABLE)
							|| blockState.is(Blocks.FLETCHING_TABLE) || blockState.is(Blocks.GRINDSTONE) || blockState.is(Blocks.LOOM)
							|| blockState.is(Blocks.SMITHING_TABLE) || blockState.is(Blocks.STONECUTTER)) {
						jobSites++;
					}

				}
			}
		}

		if (beds > 0) {
			scene.add("Villager has " + beds + " bed block(s) nearby, likely inside or near a house.");
		}
		if (doors > 0) {
			scene.add("Villager has " + doors + " door block(s) nearby.");
		}
		if (crops > 0) {
			scene.add("Villager has " + crops + " crop block(s) nearby.");
		}
		if (chests > 0) {
			scene.add("Villager has " + chests + " chest/barrel block(s) nearby.");
		}
		if (jobSites > 0) {
			scene.add("Villager has " + jobSites + " workstation block(s) nearby.");
		}

		var nearbyBox = new AABB(origin).inflate(CONTEXT_SCAN_RADIUS_BLOCKS);
		var nearbyEntities = level.getEntities((Entity) villager, nearbyBox, entity -> entity.isAlive() && entity != villager);
		long droppedItems = nearbyEntities.stream().filter(entity -> entity instanceof ItemEntity).count();
		long monsters = nearbyEntities.stream().filter(entity -> entity instanceof Monster).count();
		long otherVillagers = nearbyEntities.stream().filter(entity -> entity instanceof Villager).count();
		if (droppedItems > 0) {
			scene.add("There are " + droppedItems + " dropped item(s) near the villager.");
		}
		if (monsters > 0) {
			scene.add("Danger: hostile mobs nearby: " + summarizeEntityTypes(nearbyEntities, Monster.class) + ".");
		}
		if (otherVillagers > 0) {
			scene.add("There are " + otherVillagers + " other villager(s) nearby.");
		}

		return scene;
	}

	private String summarizeEntityTypes(List<Entity> entities, Class<?> type) {
		var counts = new HashMap<String, Integer>();
		for (var entity : entities) {
			if (!type.isInstance(entity)) {
				continue;
			}

			var key = BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType());
			var name = key == null ? entity.getType().toShortString() : key.getPath().replace('_', ' ');
			counts.merge(name, 1, Integer::sum);
		}

		var summary = new ArrayList<String>();
		for (var entry : counts.entrySet()) {
			summary.add(entry.getValue() + " " + entry.getKey());
		}
		summary.sort(String::compareTo);
		return String.join(", ", summary);
	}

	private void sendPlayerMessage(ServerPlayer minecraftPlayer, String message) {
		minecraftPlayer.sendSystemMessage(Component.literal(message));
	}

	private void playVillagerVoice(Villager villager, String line) {
		var api = voicechatApi;
		if (api == null || !villager.isAlive()) {
			return;
		}

		var channel = api.createEntityAudioChannel(UUID.randomUUID(), api.fromEntity(villager));
		if (channel == null) {
			OnaironsLivingVillagers.LOGGER.warn("Could not create villager voice audio channel");
			return;
		}

		channel.setDistance(VILLAGER_VOICE_DISTANCE_BLOCKS);
		var encoder = api.createEncoder(OpusEncoderMode.VOIP);
		var audio = VillagerMumbleSynthesizer.synthesize(line);
		var audioPlayer = api.createAudioPlayer(channel, encoder, audio);
		audioPlayer.setOnStopped(() -> {
			channel.flush();
			encoder.close();
		});
		audioPlayer.startPlaying();
	}

	private static class SpeechSegmentBuffer {
		private int packetCount;
		private int opusByteCount;
		private boolean whispering;
		private long startedAtMs;
		private long lastPacketAtMs;
		private OpusDecoder decoder;
		private ByteArrayOutputStream pcmAudio = new ByteArrayOutputStream();

		synchronized void addPacket(byte[] opusData, boolean whispering, long nowMs, VoicechatServerApi api) {
			if (packetCount == 0) {
				startedAtMs = nowMs;
				this.whispering = whispering;
			}

			packetCount++;
			opusByteCount += opusData.length;
			lastPacketAtMs = nowMs;
			this.whispering = this.whispering || whispering;
			decodePcm(opusData, api);
		}

		synchronized boolean hasAudio() {
			return packetCount > 0;
		}

		synchronized long lastPacketAtMs() {
			return lastPacketAtMs;
		}

		synchronized SpeechSegment finish() {
			var pcmBase64 = Base64.getEncoder().encodeToString(pcmAudio.toByteArray());
			var segment = new SpeechSegment(
					packetCount,
					opusByteCount,
					packetCount <= 0 ? 0L : Math.max(1L, lastPacketAtMs - startedAtMs),
					whispering,
					pcmBase64
			);

			if (decoder != null) {
				decoder.close();
			}
			packetCount = 0;
			opusByteCount = 0;
			whispering = false;
			startedAtMs = 0L;
			lastPacketAtMs = 0L;
			decoder = null;
			pcmAudio = new ByteArrayOutputStream();

			return segment;
		}

		private void decodePcm(byte[] opusData, VoicechatServerApi api) {
			if (api == null) {
				return;
			}

			try {
				if (decoder == null) {
					decoder = api.createDecoder();
				}

				var samples = decoder.decode(opusData);
				for (var sample : samples) {
					pcmAudio.write(sample & 0xFF);
					pcmAudio.write((sample >> 8) & 0xFF);
				}
			} catch (RuntimeException ignored) {
				if (decoder != null) {
					decoder.close();
					decoder = null;
				}
			}
		}
	}

	private record SpeechSegment(
			int packetCount,
			int opusByteCount,
			long durationMs,
			boolean whispering,
			String pcmAudioBase64
	) {
	}
}
