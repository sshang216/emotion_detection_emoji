from app import AppConfig, RealtimeEmojiApp


def main() -> None:
    config = AppConfig(
        camera_id=1,
        calibration_frames=45,
        ema_alpha=0.30,
    )
    RealtimeEmojiApp(config).run()


if __name__ == '__main__':
    main()
