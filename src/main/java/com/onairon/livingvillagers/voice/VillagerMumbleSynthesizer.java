package com.onairon.livingvillagers.voice;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

final class VillagerMumbleSynthesizer {
	private static final int SAMPLE_RATE = 48_000;
	private static final int FRAME_SIZE = 960;
	private static final double TWO_PI = Math.PI * 2.0D;

	private VillagerMumbleSynthesizer() {
	}

	static short[] synthesize(String line) {
		var pulses = pulsePlan(line);
		int totalSamples = 0;
		for (var pulse : pulses) {
			totalSamples += pulse.samples() + pulse.pauseSamples();
		}

		int paddedSamples = roundUpToFrame(totalSamples);
		var audio = new short[paddedSamples];
		var random = new Random(line == null ? 0 : line.hashCode());
		int cursor = 0;

		for (var pulse : pulses) {
			writePulse(audio, cursor, pulse, random);
			cursor += pulse.samples() + pulse.pauseSamples();
		}

		return audio;
	}

	private static List<Pulse> pulsePlan(String line) {
		var text = line == null || line.isBlank() ? "hrrm" : line.toLowerCase();
		var words = text.split("\\s+");
		var pulses = new ArrayList<Pulse>();
		var random = new Random(text.hashCode());

		for (var word : words) {
			var clean = word.replaceAll("[^a-z0-9]", "");
			if (clean.isBlank()) {
				continue;
			}

			int syllables = Math.max(1, Math.min(4, (clean.length() + 3) / 4));
			for (int i = 0; i < syllables; i++) {
				double annoyance = 1.0D + random.nextDouble() * 0.35D;
				double frequency = 105.0D + random.nextDouble() * 55.0D + (clean.charAt(0) % 13);
				int duration = (int) ((0.075D + random.nextDouble() * 0.055D) * SAMPLE_RATE);
				int pause = (int) ((0.014D + random.nextDouble() * 0.018D) * SAMPLE_RATE);
				pulses.add(new Pulse(duration, pause, frequency, annoyance));
			}
		}

		if (pulses.isEmpty()) {
			pulses.add(new Pulse((int) (0.18D * SAMPLE_RATE), (int) (0.03D * SAMPLE_RATE), 120.0D, 1.0D));
		}

		return pulses;
	}

	private static void writePulse(short[] audio, int start, Pulse pulse, Random random) {
		double phase = random.nextDouble() * TWO_PI;
		double wobblePhase = random.nextDouble() * TWO_PI;

		for (int i = 0; i < pulse.samples() && start + i < audio.length; i++) {
			double progress = (double) i / Math.max(1, pulse.samples() - 1);
			double attack = Math.min(1.0D, progress / 0.16D);
			double release = Math.min(1.0D, (1.0D - progress) / 0.22D);
			double envelope = Math.max(0.0D, Math.min(attack, release));
			double t = (double) i / SAMPLE_RATE;
			double wobble = Math.sin(TWO_PI * 7.5D * t + wobblePhase) * 0.055D;
			double base = pulse.frequency() * pulse.annoyance() * (1.0D + wobble);
			double wave = Math.sin(TWO_PI * base * t + phase);
			wave += 0.48D * Math.sin(TWO_PI * base * 2.02D * t + phase * 0.7D);
			wave += 0.22D * Math.sin(TWO_PI * base * 3.15D * t + phase * 1.3D);
			wave = Math.tanh(wave * 1.35D) * envelope * 0.36D;
			audio[start + i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, wave * Short.MAX_VALUE));
		}
	}

	private static int roundUpToFrame(int samples) {
		if (samples <= 0) {
			return FRAME_SIZE;
		}
		int remainder = samples % FRAME_SIZE;
		return remainder == 0 ? samples : samples + FRAME_SIZE - remainder;
	}

	private record Pulse(int samples, int pauseSamples, double frequency, double annoyance) {
	}
}
